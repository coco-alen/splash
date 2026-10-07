#pragma once

#include "metal/abi/KvZip.h"
#include "metal/abi/PagedAttention.h"
#include "metal/kernels/common/kv_extent.h"
#include "metal/kernels/common/kv_paging.h"
#include <metal_stdlib>

using namespace metal;

// ZBF16 pages (abi/KvZip.h): the store kernels encode current rows into
// their slabs, the attention kernels decode a slab into a BF16 threadgroup
// tile. Both run 256 threads per threadgroup.

// One page's bytes of keys (or values) and of their aux in one layer.
template <uint KVHeads> struct SplashKvZipBytes {
  static constexpr constant uint Data = KVHeads * SPLASH_KVZIP_DATA_BYTES_PER_HEAD;
  static constexpr constant uint Aux = KVHeads * SPLASH_KVZIP_AUX_BYTES_PER_HEAD;
};

// One KV head's slab of a page's keys or values in one layer.
struct SplashKvZipSlab {
  device uchar *data;   // sm[32][256], then codes[32][96]
  device uint *flags;   // [32][2]
  device ushort *spill; // [32]
  device uchar *slot;   // [320]
};

// A page's spill pool: the shares of every slab of the page, which lie in
// each (layer, tensor) region of its extent. Byte v of the pool is byte
// v % 128 of share v / 128.
template <uint KVHeads> struct SplashKvZipSpill {
  using Bytes = SplashKvZipBytes<KVHeads>;
  device uchar *extent;
  uint extent_pages;
  uint index;
  uint capacity;

  device uchar *byte(uint at) const {
    const uint share = at / SPLASH_KVZIP_SHARE_BYTES;
    const uint layer = share / (2u * KVHeads);
    const uint tensor = (share / KVHeads) & 1u;
    const uint head = share % KVHeads;
    return extent +
           splash_kv_offset(extent_pages, Bytes::Data, Bytes::Aux, layer,
                            tensor ? SPLASH_KV_VALUE_SCALES : SPLASH_KV_KEY_SCALES, index) +
           head * SPLASH_KVZIP_AUX_BYTES_PER_HEAD + SPLASH_KVZIP_SHARE_OFFSET +
           at % SPLASH_KVZIP_SHARE_BYTES;
  }
  // The allocation counter: the pool's first 4 bytes, in layer 0's first key
  // share.
  device atomic_uint *counter() const {
    return reinterpret_cast<device atomic_uint *>(byte(0));
  }
};

template <uint KVHeads> struct SplashKvZipAddressing {
  using Bytes = SplashKvZipBytes<KVHeads>;
  uint layer_offset;
  uint layer;
  uint extent_pages;
  ulong key_aux_offset;
  ulong values_offset;
  ulong value_aux_offset;

  explicit SplashKvZipAddressing(SplashKvLayer kv)
      : layer_offset(kv.offset),
        layer(kv.offset / uint(kv.extent_pages * 2u * (Bytes::Data + Bytes::Aux))),
        extent_pages(kv.extent_pages),
        key_aux_offset(splash_kv_offset(kv.extent_pages, Bytes::Data, Bytes::Aux, 0,
                                        SPLASH_KV_KEY_SCALES, 0)),
        values_offset(splash_kv_offset(kv.extent_pages, Bytes::Data, Bytes::Aux, 0,
                                       SPLASH_KV_VALUES, 0)),
        value_aux_offset(splash_kv_offset(kv.extent_pages, Bytes::Data, Bytes::Aux, 0,
                                          SPLASH_KV_VALUE_SCALES, 0)) {}

  SplashKvZipSlab slab(SplashKvPage entry, uint head, bool value_tensor) const {
    const uint index = splash_kv_page_index(entry);
    device uchar *region = splash_kv_extent(entry, index) + layer_offset;
    const ulong data = (value_tensor ? values_offset : 0) + ulong(index) * Bytes::Data +
                       head * SPLASH_KVZIP_DATA_BYTES_PER_HEAD;
    const ulong aux = (value_tensor ? value_aux_offset : key_aux_offset) +
                      ulong(index) * Bytes::Aux + head * SPLASH_KVZIP_AUX_BYTES_PER_HEAD;
    SplashKvZipSlab result;
    result.data = region + data;
    result.flags = reinterpret_cast<device uint *>(region + aux);
    result.spill = reinterpret_cast<device ushort *>(region + aux + SPLASH_KVZIP_SPILL_ROWS_OFFSET);
    result.slot = region + aux + SPLASH_KVZIP_SLOT_OFFSET;
    return result;
  }

  SplashKvZipSpill<KVHeads> spill(SplashKvPage entry, device const uchar *codec) const {
    const uint index = splash_kv_page_index(entry);
    const uint layers = reinterpret_cast<device const SplashKvZipHeader *>(codec)->layers;
    return {splash_kv_extent(entry, index), extent_pages, index,
            splash_kvzip_spill_bytes(layers, KVHeads)};
  }

  // The head's base3[256], followed by its base4[256], in the codec buffer.
  device const uchar *bases(device const uchar *codec, uint head, bool value_tensor) const {
    return codec + splash_kvzip_base_offset(KVHeads, layer, value_tensor ? 1u : 0u, head);
  }
};

// The overflow bytes of the groups before `groups` in a row's flag words.
inline uint splash_kvzip_row_prefix(uint word0, uint word1, uint groups) {
  if (groups <= 16)
    return splash_kvzip_flag_cost(groups == 16 ? word0 : word0 & ((1u << (2 * groups)) - 1u));
  return splash_kvzip_flag_cost(word0) +
         splash_kvzip_flag_cost(word1 & ((1u << (2 * (groups - 16))) - 1u));
}

// Encodes rows [row_begin, row_end) of one slab, in order, one thread per
// dimension. `source(row)` returns the row's element of this thread's
// dimension. Rows before row_begin are committed: their flags and spill
// marks give the slot bytes they hold. A row that does not fit the rest of
// the slot takes its bytes from the page's spill pool; one that finds that
// exhausted too loses its promoted groups and counts the slab.
// Threadgroup scratch: 32 + 8 + 1 words.
template <uint KVHeads, typename Source>
inline void splash_kvzip_store_rows(
    SplashKvZipSlab slab, SplashKvZipSpill<KVHeads> spill, device const uchar *bases,
    device atomic_uint *overflow_slabs, uint row_begin, uint row_end, Source source,
    threadgroup uint *group_tier, threadgroup uint *partial, threadgroup uint *spilled,
    uint dimension, uint simd_lane, uint simd_group) {
  const uint base3 = bases[dimension];
  const uint base4 = bases[SPLASH_KVZIP_DIMENSIONS + dimension];
  const uint lane8 = simd_lane & 7u;
  const uint group_lanes = simd_lane & ~7u;
  const uint group = dimension / SPLASH_KVZIP_GROUP;

  // The slot bytes the committed rows hold.
  uint committed = 0;
  if (dimension < row_begin && slab.spill[dimension] == SPLASH_KVZIP_IN_SLOT)
    committed = splash_kvzip_flag_cost(slab.flags[2 * dimension]) +
                splash_kvzip_flag_cost(slab.flags[2 * dimension + 1]);
  committed = simd_sum(committed);
  if (simd_lane == 0)
    partial[simd_group] = committed;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint used = 0;
  for (uint index = 0; index < 8; ++index)
    used += partial[index];
  bool lost_any = false;

  for (uint row = row_begin; row < row_end; ++row) {
    const uint bits = as_type<ushort>(source(row));
    const uint exponent = (bits >> 7) & 0xFFu;
    const uint sm = ((bits >> 8) & 0x80u) | (bits & 0x7Fu);
    const int d3 = int(exponent) - int(base3);
    const int d4 = int(exponent) - int(base4);
    const bool in3 = d3 >= 0 && d3 < 8;
    const bool in4 = d4 >= 0 && d4 < 16;
    const ulong vote3 = ulong(static_cast<simd_vote::vote_t>(simd_ballot(in3)));
    const ulong vote4 = ulong(static_cast<simd_vote::vote_t>(simd_ballot(in4)));
    const bool all3 = ((vote3 >> group_lanes) & 0xFFul) == 0xFFul;
    const bool all4 = ((vote4 >> group_lanes) & 0xFFul) == 0xFFul;
    uint tier = all3 ? SPLASH_KVZIP_TIER_WINDOW8
                     : (all4 ? SPLASH_KVZIP_TIER_WINDOW16 : SPLASH_KVZIP_TIER_RAW);
    const uint code = tier == SPLASH_KVZIP_TIER_WINDOW8
                          ? uint(d3) & 7u
                          : (tier == SPLASH_KVZIP_TIER_WINDOW16 ? uint(d4) & 7u : exponent & 7u);
    // The group's 24-bit code word and its overflow bytes, gathered by
    // disjoint ORs across the group's eight lanes.
    uint word = code << (3u * lane8);
    const ulong high = ulong(exponent >> 3) << (5u * lane8);
    uint high_low = uint(high), high_top = uint(high >> 32);
    for (ushort mask = 1; mask < 8; mask <<= 1) {
      word |= simd_shuffle_xor(word, mask);
      high_low |= simd_shuffle_xor(high_low, mask);
      high_top |= simd_shuffle_xor(high_top, mask);
    }
    const ulong vote_bit = ulong(static_cast<simd_vote::vote_t>(
        simd_ballot(in4 && ((uint(d4) >> 3) & 1u))));
    const uint window16_byte = uint((vote_bit >> group_lanes) & 0xFFul);

    if (lane8 == 0)
      group_tier[group] = tier;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint before = 0, total = 0;
    for (uint index = 0; index < SPLASH_KVZIP_GROUPS_PER_ROW; ++index) {
      const uint cost = splash_kvzip_flag_cost(group_tier[index]);
      before += index < group ? cost : 0u;
      total += cost;
    }
    // Where the row's bytes go: the slot's rest, else the spill pool.
    if (dimension == 0) {
      uint at = SPLASH_KVZIP_IN_SLOT;
      if (total && used + total > SPLASH_KVZIP_SLOT_BYTES) {
        at = atomic_fetch_add_explicit(spill.counter(), total, memory_order_relaxed);
        if (at + total > spill.capacity)
          at = SPLASH_KVZIP_IN_SLOT - 1; // exhausted: the row's groups are lost
      }
      *spilled = at;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint at = *spilled;
    const bool lost = at == SPLASH_KVZIP_IN_SLOT - 1;
    const bool in_slot = at == SPLASH_KVZIP_IN_SLOT;
    if (lost && tier != SPLASH_KVZIP_TIER_WINDOW8)
      tier = SPLASH_KVZIP_TIER_LOST;
    lost_any |= lost;

    slab.data[row * SPLASH_KVZIP_DIMENSIONS + dimension] = uchar(sm);
    if (lane8 == 0) {
      device uchar *codes = slab.data + SPLASH_KVZIP_SM_BYTES +
                            row * SPLASH_KVZIP_CODE_BYTES_PER_ROW + group * 3u;
      // A lost group keeps the code of its window-8 base, so it decodes
      // near its value.
      const uint stored = tier == SPLASH_KVZIP_TIER_LOST ? 0u : word;
      codes[0] = uchar(stored);
      codes[1] = uchar(stored >> 8);
      codes[2] = uchar(stored >> 16);
      if (tier == SPLASH_KVZIP_TIER_WINDOW16 || tier == SPLASH_KVZIP_TIER_RAW) {
        uchar bytes[5] = {uchar(window16_byte), 0, 0, 0, 0};
        if (tier == SPLASH_KVZIP_TIER_RAW) {
          bytes[0] = uchar(high_low);
          bytes[1] = uchar(high_low >> 8);
          bytes[2] = uchar(high_low >> 16);
          bytes[3] = uchar(high_low >> 24);
          bytes[4] = uchar(high_top);
        }
        const uint count = tier == SPLASH_KVZIP_TIER_RAW ? 5u : 1u;
        for (uint byte = 0; byte < count; ++byte) {
          if (in_slot)
            slab.slot[used + before + byte] = bytes[byte];
          else
            *spill.byte(at + before + byte) = bytes[byte];
        }
      }
    }
    if (lane8 == 0)
      group_tier[group] = tier;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (dimension < SPLASH_KVZIP_FLAG_WORDS_PER_ROW) {
      uint flags = 0;
      for (uint index = 0; index < 16; ++index)
        flags |= group_tier[dimension * 16 + index] << (2 * index);
      slab.flags[row * SPLASH_KVZIP_FLAG_WORDS_PER_ROW + dimension] = flags;
    }
    if (dimension == 0)
      slab.spill[row] = ushort(in_slot || lost ? SPLASH_KVZIP_IN_SLOT : at);
    used += in_slot ? total : 0u;
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (dimension == 0 && lost_any)
    atomic_fetch_add_explicit(overflow_slabs, 1u, memory_order_relaxed);
}

// One thread's 32 dimensions of a head's bases: base3 and base4 bytes,
// four to a word. Constant over a head's pages, so a tile loads them once.
struct SplashKvZipLaneBases {
  uint base3[8];
  uint base4[8];
};

inline SplashKvZipLaneBases splash_kvzip_lane_bases(device const uchar *bases, uint thread_index) {
  const uint first = (thread_index & 7u) * 32u;
  device const uint4 *base3 = reinterpret_cast<device const uint4 *>(bases + first);
  device const uint4 *base4 =
      reinterpret_cast<device const uint4 *>(bases + SPLASH_KVZIP_DIMENSIONS + first);
  SplashKvZipLaneBases result;
  for (uint side = 0; side < 2; ++side) {
    const uint4 three = base3[side], four = base4[side];
    for (uint word = 0; word < 4; ++word) {
      result.base3[4 * side + word] = three[word];
      result.base4[4 * side + word] = four[word];
    }
  }
  return result;
}

// Four elements' BF16 bits, as two words of two, from their sign/mantissa
// bytes and exponent bytes (four to a word each).
inline uint2 splash_kvzip_pack4(uint sm, uint exponent) {
  const uint high = (sm & 0x80808080u) | ((exponent >> 1) & 0x7F7F7F7Fu);
  const uint low = ((exponent & 0x01010101u) << 7) | (sm & 0x7F7F7F7Fu);
  return uint2((low & 0xFFu) | ((high & 0xFFu) << 8) | ((low & 0xFF00u) << 8) |
                   ((high & 0xFF00u) << 16),
               ((low >> 16) & 0xFFu) | (((high >> 16) & 0xFFu) << 8) |
                   ((low & 0xFF000000u) >> 8) | (high & 0xFF000000u));
}

// A group's eight decoded key elements, contiguous in a token-major slab.
inline void splash_kvzip_store_key_group(threadgroup ushort *bits, thread const uint *out) {
  *reinterpret_cast<threadgroup uint4 *>(bits) = uint4(out[0], out[1], out[2], out[3]);
}
inline void splash_kvzip_store_key_group(device ushort *bits, thread const uint *out) {
  *reinterpret_cast<device uint4 *>(bits) = uint4(out[0], out[1], out[2], out[3]);
}

// Four 3-bit codes (12 bits) as four bytes.
inline uint splash_kvzip_spread_codes(uint codes) {
  return (codes & 7u) | ((codes & 0x38u) << 5) | ((codes & 0x1C0u) << 10) |
         ((codes & 0xE00u) << 15);
}

// Decodes one slab's first `rows` rows (splash_kvzip_load) into a BF16 slab, keys token-major
// ([token][dimension]) or values dimension-major ([dimension][token]), as a
// BF16 page holds them and the paged attention tile consumes them: a
// threadgroup tile, or a BF16 page's slab in device memory. Rows past `rows`
// become zeros. Thread t decodes row t / 8, dimensions 32 (t % 8) on, four
// elements per word. Scratch: 8 words. Ends with a threadgroup barrier.
// One thread's share of a slab, loaded ahead of its decode so the loads
// overlap other work: its row's flags and spill mark, its 32 sign/mantissa
// bytes and 12 code bytes, and one word of the slot.
struct SplashKvZipLoaded {
  uint4 sm_low, sm_high;
  uint codes[3];
  uint flags[2];
  uint spill;
  uint slot;
  bool valid;
};

inline SplashKvZipLoaded splash_kvzip_load(SplashKvZipSlab slab, uint rows, uint thread_index) {
  constexpr uint D = SPLASH_KVZIP_DIMENSIONS;
  const uint row = thread_index >> 3;
  const uint part = thread_index & 7u;
  SplashKvZipLoaded loaded{};
  loaded.valid = row < rows;
  loaded.spill = SPLASH_KVZIP_IN_SLOT;
  if (loaded.valid) {
    const uint2 words = reinterpret_cast<device const uint2 *>(slab.flags)[row];
    loaded.flags[0] = words.x, loaded.flags[1] = words.y;
    loaded.spill = slab.spill[row];
    device const uint *codes = reinterpret_cast<device const uint *>(
        slab.data + SPLASH_KVZIP_SM_BYTES + row * SPLASH_KVZIP_CODE_BYTES_PER_ROW + part * 12u);
    loaded.codes[0] = codes[0], loaded.codes[1] = codes[1], loaded.codes[2] = codes[2];
    device const uint4 *sm = reinterpret_cast<device const uint4 *>(slab.data + row * D + part * 32u);
    loaded.sm_low = sm[0], loaded.sm_high = sm[1];
  }
  if (thread_index < SPLASH_KVZIP_SLOT_BYTES / 4)
    loaded.slot = reinterpret_cast<device const uint *>(slab.slot)[thread_index];
  return loaded;
}

template <uint KVHeads, typename Bits>
inline void splash_kvzip_decode_slab(thread const SplashKvZipLoaded &loaded,
                                     SplashKvZipSpill<KVHeads> spill,
                                     thread const SplashKvZipLaneBases &bases,
                                     bool value_tensor, Bits bits, threadgroup uint *partial,
                                     threadgroup uint *slot_words, uint thread_index,
                                     uint simd_lane, uint simd_group) {
  constexpr uint D = SPLASH_KVZIP_DIMENSIONS;
  constexpr uint N = SPLASH_KVZIP_ROWS;
  const uint row = thread_index >> 3;
  const uint part = thread_index & 7u;
  const bool valid = loaded.valid;
  const uint word0 = loaded.flags[0], word1 = loaded.flags[1], at = loaded.spill;
  const bool in_slot = at == SPLASH_KVZIP_IN_SLOT;
  const uint flags = ((part < 4 ? word0 : word1) >> (8u * (part & 3u))) & 0xFFu;
  const uint cost = splash_kvzip_flag_cost(flags);
  // Exclusive prefix of the slot rows' costs in thread order, which is (row,
  // group) order: this thread's first slot byte.
  const uint slot_cost = in_slot ? cost : 0u;
  const uint inclusive = simd_prefix_inclusive_sum(slot_cost);
  if (simd_lane == 31)
    partial[simd_group] = inclusive;
  // The slot's bytes, read once and coalesced: the promoted groups' bytes
  // come from threadgroup memory (two words past them are read and unused).
  if (thread_index < SPLASH_KVZIP_SLOT_BYTES / 4)
    slot_words[thread_index] = loaded.slot;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint offset = inclusive - slot_cost;
  for (uint index = 0; index < simd_group; ++index)
    offset += partial[index];
  if (!in_slot)
    offset = at + splash_kvzip_row_prefix(word0, word1, part * 4u);

  const uint first = part * 32u;
  const uint c0 = loaded.codes[0], c1 = loaded.codes[1], c2 = loaded.codes[2];
  const uint4 sm_low = loaded.sm_low, sm_high = loaded.sm_high;
  // Each group's first overflow byte, where its bytes start: window-16
  // groups need only that byte, raw groups (rare) read theirs again below.
  uint group_offset[4], first_byte[4];
#pragma unroll
  for (uint group = 0; group < 4; ++group) {
    const uint tier = (flags >> (2u * group)) & 3u;
    group_offset[group] = offset;
    if (in_slot) {
      const uint word = offset >> 2, shift = 8u * (offset & 3u);
      const uint low = slot_words[word], high = slot_words[word + 1];
      first_byte[group] = (shift ? (low >> shift) | (high << (32u - shift)) : low) & 0xFFu;
    } else {
      first_byte[group] = tier ? uint(*spill.byte(offset)) : 0u;
    }
    offset += tier == SPLASH_KVZIP_TIER_RAW ? 5u : (tier == SPLASH_KVZIP_TIER_WINDOW16 ? 1u : 0u);
  }
  const uint group_codes[4] = {c0 & 0xFFFFFFu, (c0 >> 24) | ((c1 & 0xFFFFu) << 8),
                               (c1 >> 16) | ((c2 & 0xFFu) << 16), c2 >> 8};
  const uint sm_words[8] = {sm_low.x,  sm_low.y,  sm_low.z,  sm_low.w,
                            sm_high.x, sm_high.y, sm_high.z, sm_high.w};
#pragma unroll
  for (uint group = 0; group < 4; ++group) {
    const uint tier = (flags >> (2u * group)) & 3u;
    uint out[4];
#pragma unroll
    for (uint side = 0; side < 2; ++side) {
      const uint word = 2 * group + side;
      const uint code4 = splash_kvzip_spread_codes(group_codes[group] >> (12u * side));
      // Window-8 and window-16 groups without a branch: base and bit 3 by
      // the tier (the multiply spreads four bits to the bytes' low bits).
      const bool wide = tier == SPLASH_KVZIP_TIER_WINDOW16;
      const uint bit = wide ? (first_byte[group] >> (4u * side)) & 0xFu : 0u;
      uint exponent = (wide ? bases.base4[word] : bases.base3[word]) + code4 +
                      (((bit * 0x00204081u) & 0x01010101u) << 3);
      if (tier >= SPLASH_KVZIP_TIER_LOST) { // raw (0.1% of real groups) or lost
        if (tier == SPLASH_KVZIP_TIER_RAW) {
          // The group's 40 bits; this side's four 5-bit fields start at bit 20 side.
          uint fields = 0;
          for (uint byte = 0; byte < 5; ++byte) {
            const uint at = group_offset[group] + byte;
            const uint value = in_slot ? (slot_words[at >> 2] >> (8u * (at & 3u))) & 0xFFu
                                       : uint(*spill.byte(at));
            const int position = int(8u * byte) - int(20u * side);
            if (position > -8 && position < 32)
              fields |= position >= 0 ? value << position : value >> -position;
          }
          exponent = code4;
          for (uint lane = 0; lane < 4; ++lane)
            exponent |= ((fields >> (5u * lane)) & 31u) << (8u * lane + 3u);
        } else { // lost: near the value, and never infinite or NaN
          exponent = 0;
          for (uint lane = 0; lane < 4; ++lane)
            exponent |= min(((bases.base3[word] >> (8u * lane)) & 0xFFu) +
                                ((code4 >> (8u * lane)) & 0xFFu),
                            254u)
                        << (8u * lane);
        }
      }
      const uint2 packed = valid ? splash_kvzip_pack4(sm_words[word], exponent) : uint2(0);
      out[2 * side] = packed.x;
      out[2 * side + 1] = packed.y;
    }
    const uint element = first + 8 * group;
    if (value_tensor) {
#pragma unroll
      for (uint word = 0; word < 4; ++word) {
        bits[(element + 2 * word) * N + row] = ushort(out[word]);
        bits[(element + 2 * word + 1) * N + row] = ushort(out[word] >> 16);
      }
    } else {
      splash_kvzip_store_key_group(bits + row * D + element, out);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
}

// Resets the spill pool of every page a chunk starts: those whose first
// token is at or past the committed tokens. One thread per page the chunk
// touches.
template <uint KVHeads>
inline void splash_kvzip_reset_chunk_spill(device const SplashKvPage *page_table,
                                           constant SplashChunkedPrefillParams &params,
                                           uint page_offset) {
  const uint first_page = params.committed_tokens / SplashKvPageTokens;
  const uint last_page = (params.committed_tokens + params.chunk_tokens - 1) / SplashKvPageTokens;
  const uint page = first_page + page_offset;
  if (page > last_page || page * SplashKvPageTokens < params.committed_tokens)
    return;
  const SplashKvZipAddressing<KVHeads> addressing(params.kv);
  const SplashKvPage entry = page_table[page];
  const uint index = splash_kv_page_index(entry);
  SplashKvZipSpill<KVHeads> spill{splash_kv_extent(entry, index), addressing.extent_pages, index, 0};
  atomic_store_explicit(spill.counter(), SPLASH_KVZIP_SPILL_START, memory_order_relaxed);
}

// Expands one slab of a chunk's committed history into the BF16 scratch:
// threadgroup `slab` of 2 x KVHeads x history pages, keys first.
template <uint KVHeads>
inline void splash_kvzip_expand_history(device const SplashKvPage *page_table,
                                        device const SplashKvPage *scratch_table,
                                        device const uchar *codec,
                                        constant SplashKvZipExpandParams &params, uint slab_index,
                                        threadgroup uint *partial, threadgroup uint *slot_words,
                                        uint thread_index,
                                        uint simd_lane, uint simd_group) {
  const uint committed = params.chunk.committed_tokens;
  const uint pages = (committed + SplashKvPageTokens - 1) / SplashKvPageTokens;
  if (slab_index >= 2 * KVHeads * pages)
    return;
  const bool value_tensor = slab_index >= KVHeads * pages;
  const uint rest = value_tensor ? slab_index - KVHeads * pages : slab_index;
  const uint head = rest / pages;
  const uint page = rest % pages;
  const SplashKvZipAddressing<KVHeads> addressing(params.chunk.kv);
  const SplashKvPage entry = page_table[page];
  const SplashKvPageTensors<bfloat> target =
      SplashKvAddressing<KVHeads, bfloat>(params.scratch, head).page(scratch_table[page]);
  device ushort *bits =
      reinterpret_cast<device ushort *>(value_tensor ? target.values : target.keys);
  const SplashKvZipLaneBases bases =
      splash_kvzip_lane_bases(addressing.bases(codec, head, value_tensor), thread_index);
  const SplashKvZipLoaded loaded =
      splash_kvzip_load(addressing.slab(entry, head, value_tensor),
                        min(uint(SplashKvPageTokens), committed - page * SplashKvPageTokens),
                        thread_index);
  splash_kvzip_decode_slab<KVHeads>(loaded, addressing.spill(entry, codec), bases, value_tensor,
                                    bits, partial, slot_words, thread_index, simd_lane,
                                    simd_group);
}
