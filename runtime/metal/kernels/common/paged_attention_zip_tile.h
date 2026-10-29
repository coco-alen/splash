#pragma once

#include "metal/kernels/common/kv_zip.h"
#include "metal/kernels/common/paged_attention_tile.h"
#include "metal/kernels/common/paged_store_row.h"

// The paged attention tile over ZBF16 pages. It runs the BF16 tile's page
// loop (splash_paged_attention_tile) with the same MPP products, softmax and
// partials; each page's keys, then its values, are first decoded into one
// BF16 threadgroup tile, token-major, so a ZBF16 page attends exactly as the
// BF16 page it decodes to (PV reads the values untransposed, bit for bit as
// the BF16 loop's transposed read). With OwnScores the page's scores have
// their own scratch, as the BF16 loop's do, so a page's values decode beside
// its softmax, which touches neither: a page takes four barriers, after its
// keys' decode, before its softmax, after its values' decode and after its
// products. Without it the scores share the tile, written after the keys'
// last read and read before the values' first write, at two barriers more:
// 6% faster on the 27B's six query heads per KV head (M5 Pro, 2K-128K),
// within 1% on the 35B's eight, whose 8 KiB of scores cost the short
// contexts more. Threadgroup scratch beyond the BF16 loop's probabilities
// and statistics: the 16 KiB tile, and the scores with OwnScores.
// A lane's share of PV's right input in a simdgroup's 32 dimensions, the
// cooperative tensor's elements in groups of four: where each group holds
// four consecutive dimensions of one token (Apple10 lays the right input out
// so), the group's token, first dimension and those dimensions' bases,
// found once per tile; otherwise `words` is false and the values decode one
// element at a time.
struct SplashKvZipValueLane {
  static constexpr constant uint Groups = 8;
  ushort token[Groups];
  ushort dimension[Groups];
  uint base[Groups];
  bool words;
};

template <class Values>
inline SplashKvZipValueLane splash_kvzip_value_lane(thread const Values &values, uint first,
                                                    device const uchar *bases) {
  SplashKvZipValueLane lane;
  lane.words = values.get_capacity() == 4 * SplashKvZipValueLane::Groups;
  for (ushort group = 0; group < SplashKvZipValueLane::Groups; ++group) {
    const auto at = values.get_multidimensional_index(ushort(4 * group));
    lane.token[group] = ushort(at[1]);
    lane.dimension[group] = ushort(first + uint(at[0]));
    lane.words = lane.words && (lane.dimension[group] & 3u) == 0;
    for (ushort next = 1; next < 4; ++next) {
      const auto later = values.get_multidimensional_index(ushort(4 * group + next));
      lane.words = lane.words && ushort(later[1]) == lane.token[group] &&
                   first + uint(later[0]) == uint(lane.dimension[group]) + next;
    }
    lane.base[group] = lane.words ? *reinterpret_cast<device const uint *>(bases + lane.dimension[group]) : 0u;
  }
  return lane;
}

// Decodes a simdgroup's share of PV's right input from the slab's bytes into
// `decoded`, BF16 bits in the cooperative tensor's element order, straight
// into the registers the product reads, as GAAC decodes into its MMA
// operands: no threadgroup tile and no barrier. Each group of four decodes
// as the tile decode's words do, from one word of sign/mantissa bytes and two
// bytes of codes, every group's bytes loaded before the first decodes.
// Tokens past `rows` decode to zeros, as the tile's decode writes them.
template <class Values>
inline void splash_kvzip_fill_values(thread const Values &values, thread const SplashKvZipValueLane &lane,
                                     thread ushort *decoded, SplashKvZipSlab slab, uint rows, uint first,
                                     device const uchar *bases) {
  constexpr uint D = SPLASH_KVZIP_DIMENSIONS;
  constexpr uint G = SplashKvZipValueLane::Groups;
  const uint count = min(*slab.count, SPLASH_KVZIP_ESCAPES);
  if (lane.words) {
    uint sm[G], nibbles[G];
#pragma unroll
    for (uint group = 0; group < G; ++group) {
      const uint token = min(uint(lane.token[group]), rows - 1u), dimension = lane.dimension[group];
      sm[group] = *reinterpret_cast<device const uint *>(slab.data + token * D + dimension);
      nibbles[group] = *reinterpret_cast<device const ushort *>(
          slab.data + SPLASH_KVZIP_SM_BYTES + token * SPLASH_KVZIP_CODE_BYTES_PER_ROW + dimension / 2u);
    }
#pragma unroll
    for (uint group = 0; group < G; ++group) {
      const uint n = nibbles[group];
      const uint code4 = (n & 0xFu) | ((n & 0xF0u) << 4) | ((n & 0xF00u) << 8) | ((n & 0xF000u) << 12);
      const uint2 packed = lane.token[group] < rows ? splash_kvzip_pack4(sm[group], lane.base[group] + code4)
                                                    : uint2(0);
      decoded[4 * group] = ushort(packed.x);
      decoded[4 * group + 1] = ushort(packed.x >> 16);
      decoded[4 * group + 2] = ushort(packed.y);
      decoded[4 * group + 3] = ushort(packed.y >> 16);
    }
    for (uint escape = 0; escape < count; ++escape) {
      const uint entry = slab.escapes[escape], position = entry & 0xFFFFu;
      const uint token = position / D, dimension = position % D;
      if (token >= rows)
        continue;
#pragma unroll
      for (uint group = 0; group < G; ++group) {
        const uint offset = dimension - lane.dimension[group];
        if (token == lane.token[group] && offset < 4u)
          decoded[4 * group + offset] = ushort((decoded[4 * group + offset] & ~0x7F80u) | (((entry >> 16) & 0xFFu) << 7));
      }
    }
    return;
  }
  for (ushort element = 0; element < values.get_capacity(); ++element) {
    const auto here = values.get_multidimensional_index(element);
    const uint d = first + uint(here[0]), t = uint(here[1]);
    uint bits = 0;
    if (t < rows) {
      const uint sm = slab.data[t * D + d];
      const uint code = (slab.data[SPLASH_KVZIP_SM_BYTES + t * SPLASH_KVZIP_CODE_BYTES_PER_ROW + d / 2u] >>
                         (4u * (d & 1u))) & 0xFu;
      uint exponent = bases[d] + code;
      for (uint escape = 0; escape < count; ++escape) {
        const uint entry = slab.escapes[escape];
        if ((entry & 0xFFFFu) == t * D + d)
          exponent = (entry >> 16) & 0xFFu;
      }
      bits = ((sm & 0x80u) << 8) | (exponent << 7) | (sm & 0x7Fu);
    }
    decoded[element] = ushort(bits);
  }
}

template <uint KVHeads, uint QueryHeadsPerKVHead, uint RowsPerTile, bool OwnScores>
inline void splash_paged_attention_zip_tile(
    device bfloat *tile_queries, device const SplashKvPage *page_table,
    SplashKvLayer kv, device const uchar *codec, uint kv_head, uint committed_tokens,
    uint active_rows, uint splits, uint split, device float *partials,
    device float *statistics, ulong slot, threadgroup float *scores, threadgroup bfloat *probabilities,
    threadgroup float *row_max, threadgroup float *row_sum, threadgroup float *previous_scale,
    threadgroup atomic_uint *rescale, threadgroup bfloat *kv_tile, uint thread_index) {
  constexpr ushort M = RowsPerTile * QueryHeadsPerKVHead;
  constexpr ushort N = SplashKvPageTokens;
  constexpr ushort D = SplashKvHeadDimension;
  if (!OwnScores)
    scores = reinterpret_cast<threadgroup float *>(kv_tile);
  threadgroup ushort *bits = reinterpret_cast<threadgroup ushort *>(kv_tile);
  uint visible_tokens = committed_tokens + active_rows;
  uint pages = splash_attention_pages(visible_tokens);
  uint per_split = splash_attention_pages_per_split(pages, splits);
  uint page_begin = split * per_split;
  if (page_begin >= pages)
    return;
  uint page_end = min(pages, page_begin + per_split);
  if (thread_index < M) {
    row_max[thread_index] = -INFINITY;
    row_sum[thread_index] = 0.0f;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  auto qt = tensor(tile_queries, dextents<int, 2>{D, M}, array<int, 2>{1, D});
  auto st = tensor(scores, dextents<int, 2>{N, M}, array<int, 2>{1, N});
  auto pt = tensor(probabilities, dextents<int, 2>{N, M}, array<int, 2>{1, N});
  auto kt = tensor(kv_tile, dextents<int, 2>{D, N}, array<int, 2>{1, D});
  auto p0 = pt.slice<N, M>(0, 0);
  auto q0 = qt.slice<D, M>(0, 0);
  auto k0 = kt.slice<D, N>(0, 0);
  const SplashKvZipAddressing<KVHeads> addressing(kv);
  const SplashKvZipLaneBases key_bases =
      splash_kvzip_lane_bases(addressing.bases(codec, kv_head, false), thread_index);
  device const uchar *value_bases = addressing.bases(codec, kv_head, true);
  // PV runs per simdgroup over its own 32 output dimensions, whose values it
  // decodes into the product's right input (splash_kvzip_fill_values).
  constexpr ushort Columns = D / 8;
  const uint simd_group = thread_index / 32u;
  constexpr auto qk_descriptor =
      matmul2d_descriptor(M, N, D, false, true, false,
                          matmul2d_descriptor::mode::multiply);
  constexpr auto pv_descriptor =
      matmul2d_descriptor(M, Columns, N, false, false, true,
                          matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<qk_descriptor, execution_simdgroups<8>> qk;
  matmul2d<pv_descriptor, execution_simdgroup> pv;
  auto page_values = pv.template get_right_input_cooperative_tensor<bfloat, bfloat, float>();
  auto running = pv.template get_destination_cooperative_tensor<
      decltype(p0), decltype(page_values), float>();
  const SplashKvZipValueLane value_lane =
      splash_kvzip_value_lane(page_values, simd_group * Columns, value_bases);
  const bool running_full = uint(running.get_capacity()) * 32u == uint(M) * Columns;
#pragma unroll
  for (ushort index = 0; index < running.get_capacity(); ++index) {
    if (running_full || running.is_valid_element(index))
      running[index] = 0.0f;
  }

  for (uint page = page_begin; page < page_end; ++page) {
    const SplashKvPage entry = page_table[page];
    uint token_start = page * N;
    // Rows past the visible tokens hold stale bytes: they decode to zeros,
    // which the softmax masks as the BF16 loop masks its stale rows.
    const uint rows = min(uint(N), visible_tokens - token_start);
    const SplashKvZipSlab keys = addressing.slab(entry, kv_head, false);
    const SplashKvZipSlab values = addressing.slab(entry, kv_head, true);
    splash_kvzip_decode_slab(splash_kvzip_load(keys, rows, thread_index), keys, key_bases, false,
                             bits, thread_index);
    auto page_scores = qk.template get_destination_cooperative_tensor<
        decltype(q0), decltype(k0), float>();
    qk.run(q0, k0, page_scores);
    // Every simdgroup has read the keys before the scores overwrite them.
    if (!OwnScores)
      threadgroup_barrier(mem_flags::mem_threadgroup);
    page_scores.store(st.slice<N, M>(0, 0));
    if (thread_index == 0)
      atomic_store_explicit(rescale, 0u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    splash_attention_page_softmax<QueryHeadsPerKVHead, RowsPerTile, false>(
        scores, probabilities, row_max, row_sum, previous_scale, rescale, nullptr,
        nullptr, token_start, visible_tokens, committed_tokens, active_rows,
        thread_index);
    {
      ushort decoded[Columns * N / 32];
      splash_kvzip_fill_values(page_values, value_lane, decoded, values, rows, simd_group * Columns,
                               value_bases);
#pragma unroll
      for (ushort i = 0; i < page_values.get_capacity(); ++i)
        page_values[i] = as_type<bfloat>(decoded[i]);
    }
    // The softmax's probabilities and scales, before the products read them.
    // The next page's keys overwrite the tile after its decode's barrier, so
    // the products need none after them.
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (atomic_load_explicit(rescale, memory_order_relaxed)) {
#pragma unroll
      for (ushort index = 0; index < running.get_capacity(); ++index) {
        if (!running_full && !running.is_valid_element(index))
          continue;
        auto coordinates = running.get_multidimensional_index(index);
        running[index] *= previous_scale[coordinates[1]];
      }
    }
    pv.run(p0, page_values, running);
  }
  auto target = tensor(partials + slot * M * D, dextents<int, 2>{D, M},
                       array<int, 2>{1, D});
  running.store(target.slice<Columns, M>(simd_group * Columns, 0));
  if (thread_index < M) {
    statistics[(slot * M + thread_index) * 2] = row_max[thread_index];
    statistics[(slot * M + thread_index) * 2 + 1] = row_sum[thread_index];
  }
}

// A chunk row's element of one dimension, by the slab's row.
struct SplashKvZipChunkRows {
  device const bfloat *chunk_keys;
  device const bfloat *chunk_values;
  uint stride;
  uint head;
  uint chunk_offset;
  uint dimension;
  bool value_tensor;

  bfloat operator()(uint row) const {
    const uint chunk_token = chunk_offset + row;
    return value_tensor
               ? chunk_values[splash_current_value_index(stride, head, chunk_token, dimension)]
               : chunk_keys[splash_current_key_index(stride, head, chunk_token, dimension)];
  }
};

// Stores a chunk's (or a verify lane's) current rows: threadgroup `slab`
// of 2 x KVHeads x pages, keys first, encodes one KV head's rows of one page
// in order. `pages` is the number of pages the rows touch.
template <uint KVHeads>
inline void splash_kvzip_store_chunk_slab(
    device const bfloat *chunk_keys, device const bfloat *chunk_values,
    device const SplashKvPage *page_table, device uchar *codec,
    constant SplashChunkedPrefillParams &params, uint slab_index,
    threadgroup uint *partial, uint dimension, uint simd_lane, uint simd_group) {
  const uint first = params.committed_tokens;
  const uint last = params.committed_tokens + params.chunk_tokens - 1;
  const uint first_page = first / SplashKvPageTokens;
  const uint pages = last / SplashKvPageTokens - first_page + 1;
  if (slab_index >= 2 * KVHeads * pages)
    return;
  const bool value_tensor = slab_index >= KVHeads * pages;
  const uint rest = value_tensor ? slab_index - KVHeads * pages : slab_index;
  const uint head = rest / pages;
  const uint page = first_page + rest % pages;
  const uint page_first = page * SplashKvPageTokens;
  const uint row_begin = max(first, page_first) - page_first;
  const uint row_end = min(last + 1, page_first + SplashKvPageTokens) - page_first;
  const SplashKvZipAddressing<KVHeads> addressing(params.kv);
  const SplashKvZipSlab slab = addressing.slab(page_table[page], head, value_tensor);
  const SplashKvZipChunkRows source{chunk_keys, chunk_values, params.chunk_stride, head,
                                    page_first - first, dimension, value_tensor};
  splash_kvzip_store_rows(slab, addressing.bases(codec, head, value_tensor),
                          reinterpret_cast<device atomic_uint *>(codec), row_begin, row_end,
                          source, partial, dimension, simd_lane, simd_group);
}
