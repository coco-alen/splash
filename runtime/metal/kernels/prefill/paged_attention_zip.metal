#include "metal/kernels/common/paged_attention_zip_tile.h"

// ZBF16 (abi/KvZip.h) prefill store and split. The reduce is the BF16 one:
// partials and statistics do not depend on the page format.

// Store: one threadgroup per (tensor, KV head, page the chunk touches)
// encodes that page's chunk rows in order.
#define PAGED_PREFILL_ZIP_STORE(Name, Heads)                                   \
  kernel void Name(                                                            \
      device const bfloat *chunk_keys [[buffer(0)]],                           \
      device const bfloat *chunk_values [[buffer(1)]],                         \
      device const SplashKvPage *page_table [[buffer(2)]],                     \
      device uchar *codec [[buffer(3)]],                                       \
      constant SplashChunkedPrefillParams &params [[buffer(4)]],               \
      uint group [[threadgroup_position_in_grid]],                             \
      uint thread_index [[thread_index_in_threadgroup]],                       \
      uint simd_lane [[thread_index_in_simdgroup]],                            \
      uint simd_group [[simdgroup_index_in_threadgroup]]) {                    \
    threadgroup uint group_tier[SPLASH_KVZIP_GROUPS_PER_ROW];                  \
    threadgroup uint partial[8];                                               \
    threadgroup uint spilled;                                                  \
    if (!splash_chunk_contract_valid(params) ||                                \
        thread_index >= SplashKvHeadDimension)                                 \
      return;                                                                  \
    splash_kvzip_store_chunk_slab<Heads>(chunk_keys, chunk_values, page_table, \
                                         codec, params, group, group_tier,     \
                                         partial, &spilled, thread_index,      \
                                         simd_lane, simd_group);               \
  }
PAGED_PREFILL_ZIP_STORE(prefill_attention_zip_store, 4)
PAGED_PREFILL_ZIP_STORE(prefill_attention_zip_store_kv2_g8, 2)
#undef PAGED_PREFILL_ZIP_STORE

// Before layer 0's store: resets the spill pool of each page the chunk
// starts, one thread per page it touches.
#define PAGED_PREFILL_ZIP_RESET(Name, Heads)                                   \
  kernel void Name(device const SplashKvPage *page_table [[buffer(0)]],        \
                   constant SplashChunkedPrefillParams &params [[buffer(1)]],  \
                   uint page [[thread_position_in_grid]]) {                    \
    if (!splash_chunk_contract_valid(params))                                  \
      return;                                                                  \
    splash_kvzip_reset_chunk_spill<Heads>(page_table, params, page);           \
  }
PAGED_PREFILL_ZIP_RESET(prefill_attention_zip_reset, 4)
PAGED_PREFILL_ZIP_RESET(prefill_attention_zip_reset_kv2_g8, 2)
#undef PAGED_PREFILL_ZIP_RESET

// Split: the BF16 prefill split's grid and slots over decoded pages.
#define PAGED_PREFILL_ZIP_SPLIT(Name, Heads, Group)                            \
  kernel void Name(                                                            \
      device bfloat *queries [[buffer(0)]],                                    \
      device float *partials [[buffer(1)]],                                    \
      device float *statistics [[buffer(2)]],                                  \
      device const SplashKvPage *page_table [[buffer(3)]],                     \
      device const uchar *codec [[buffer(4)]],                                 \
      constant SplashPrefillAttentionParams &params [[buffer(5)]],             \
      uint3 group [[threadgroup_position_in_grid]],                            \
      uint thread_index [[thread_index_in_threadgroup]],                       \
      uint simd_lane [[thread_index_in_simdgroup]],                            \
      uint simd_group [[simdgroup_index_in_threadgroup]]) {                    \
    constexpr uint M = Group * SPLASH_PREFILL_ATTENTION_TILE_ROWS;             \
    constexpr uint N = SplashKvPageTokens;                                     \
    constexpr uint D = SplashKvHeadDimension;                                  \
    alignas(16) threadgroup float scores[M * N];                               \
    alignas(16) threadgroup bfloat probabilities[M * N];                       \
    alignas(16) threadgroup bfloat kv_tile[N * D];                             \
    threadgroup float row_max[M];                                              \
    threadgroup float row_sum[M];                                              \
    threadgroup float previous_scale[M];                                       \
    threadgroup atomic_uint rescale;                                           \
    threadgroup uint partial[8];                                               \
    uint kv_head = group.x;                                                    \
    uint tile = group.y;                                                       \
    uint split = group.z;                                                      \
    uint tile_start = tile * SplashPrefillTileRows;                            \
    if (!splash_prefill_attention_contract_valid(params) ||                    \
        kv_head >= Heads || split >= params.split_count ||                     \
        tile_start >= params.rows)                                             \
      return;                                                                  \
    uint active_rows = min(SplashPrefillTileRows, params.rows - tile_start);   \
    ulong tile_offset = (ulong(kv_head) * params.chunk_stride + tile_start) *  \
                        Group * D;                                             \
    ulong slot = (ulong(tile) * Heads + kv_head) * params.split_count + split; \
    splash_paged_attention_zip_tile<Heads, Group,                              \
                                    SPLASH_PREFILL_ATTENTION_TILE_ROWS>(       \
        queries + tile_offset, page_table, params.kv, codec, kv_head,          \
        params.committed_tokens + tile_start, active_rows, params.split_count, \
        split, partials, statistics, slot, scores, probabilities, row_max,     \
        row_sum, previous_scale, &rescale, kv_tile, partial, thread_index,     \
        simd_lane, simd_group);                                                \
  }
PAGED_PREFILL_ZIP_SPLIT(prefill_attention_zip_split, 4, 6)
PAGED_PREFILL_ZIP_SPLIT(prefill_attention_zip_split_kv2_g8, 2, 8)
#undef PAGED_PREFILL_ZIP_SPLIT
