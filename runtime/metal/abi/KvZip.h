#pragma once

// ZBF16: lossless BF16 KV pages (NPLG, nested-precision lane-group coding).
// Every stored bit decodes back to the BF16 value the projection wrote; a
// page is smaller than BF16 because the exponents of one (layer, KV head,
// dimension) sit in a narrow window, which a 3-bit code covers.
//
// A page keeps the BF16 extent geometry (abi/KvExtent.h): the data region of
// keys and of values holds each KV head's sign/mantissa bytes and 3-bit
// exponent codes, and the "scale" region holds each head's group flags and
// overflow bytes. Keys and values are both stored token-major here; the
// attention kernels decode a page into the BF16 tile layout they consume.
//
// Per (page, layer, tensor, KV head) slab of 32 tokens x 256 dimensions:
//   data: sm[32][256]     sign << 7 | 7 mantissa bits, one byte per element
//         codes[32][96]   per token row, 32 groups of 8 dimensions, each a
//                         24-bit little-endian word of eight 3-bit codes
//   aux:  flags[32][2]    per row, 2 bits per group (uint32 words, group g
//                         in word g / 16 at bit 2 * (g % 16)):
//                         0 = window-8 tier:  e = base3 + code
//                         1 = window-16 tier: e = base4 + (code | bit << 3),
//                             one overflow byte holds the 8 elements' bit 3
//                         3 = raw tier:       e = code | high5 << 3, five
//                             overflow bytes hold the 8 high-5-bit fields
//                             (40 bits, little-endian)
//                         2 = lost: the page ran out of overflow space; the
//                             group decodes approximately, e = base3 + code
//         spill[32]       per row, uint16: where the row's overflow bytes
//                         sit in the page's spill pool, or 0xFFFF when they
//                         sit in the slot
//         slot[320]       the overflow bytes of the slab's rows that fit, in
//                         (row, group) order: a group's offset is the
//                         overflow of every slot row and group before it
//         share[128]      this slab's share of the page's spill pool
// A row whose overflow does not fit the slot's rest takes it from the spill
// pool, which every slab of the page shares: the shares of all its layers,
// tensors and heads in that order, 128 bytes each, with the pool's
// allocation counter in its first 4 bytes. A store resets the counter of a
// page it starts. Real KV fits the slots (Qwen3.8-27B: 99.9% of slabs under
// 233 bytes); repetitive prompts spill a few hundred bytes per page.
// base3 and base4 are per (layer, tensor, KV head, dimension) windows,
// calibrated per model family and held in the codec buffer.
//
// A row that finds the spill pool exhausted too is stored with its promoted
// groups lost; the store counts its slab in the codec buffer's overflow
// counter, which the runtime reports and logs.
#ifdef __METAL_VERSION__
#include <metal_stdlib>
#else
#include <stdint.h>
#endif

#include "metal/abi/PagedAttention.h"

#define SPLASH_KVZIP_ROWS 32u
#define SPLASH_KVZIP_DIMENSIONS 256u
#define SPLASH_KVZIP_GROUP 8u
#define SPLASH_KVZIP_GROUPS_PER_ROW (SPLASH_KVZIP_DIMENSIONS / SPLASH_KVZIP_GROUP)
#define SPLASH_KVZIP_CODE_BYTES_PER_ROW (SPLASH_KVZIP_GROUPS_PER_ROW * 3u)
#define SPLASH_KVZIP_FLAG_WORDS_PER_ROW (SPLASH_KVZIP_GROUPS_PER_ROW / 16u)

#define SPLASH_KVZIP_SM_BYTES (SPLASH_KVZIP_ROWS * SPLASH_KVZIP_DIMENSIONS)
#define SPLASH_KVZIP_CODE_BYTES (SPLASH_KVZIP_ROWS * SPLASH_KVZIP_CODE_BYTES_PER_ROW)
#define SPLASH_KVZIP_DATA_BYTES_PER_HEAD (SPLASH_KVZIP_SM_BYTES + SPLASH_KVZIP_CODE_BYTES)

#define SPLASH_KVZIP_FLAG_BYTES (SPLASH_KVZIP_ROWS * SPLASH_KVZIP_FLAG_WORDS_PER_ROW * 4u)
#define SPLASH_KVZIP_SPILL_ROW_BYTES (SPLASH_KVZIP_ROWS * 2u)
#define SPLASH_KVZIP_SLOT_BYTES 320u
#define SPLASH_KVZIP_SHARE_BYTES 128u
#define SPLASH_KVZIP_SPILL_ROWS_OFFSET SPLASH_KVZIP_FLAG_BYTES
#define SPLASH_KVZIP_SLOT_OFFSET (SPLASH_KVZIP_SPILL_ROWS_OFFSET + SPLASH_KVZIP_SPILL_ROW_BYTES)
#define SPLASH_KVZIP_SHARE_OFFSET (SPLASH_KVZIP_SLOT_OFFSET + SPLASH_KVZIP_SLOT_BYTES)
#define SPLASH_KVZIP_AUX_BYTES_PER_HEAD (SPLASH_KVZIP_SHARE_OFFSET + SPLASH_KVZIP_SHARE_BYTES)
// A row in the slot; the spill pool's first bytes hold its counter.
#define SPLASH_KVZIP_IN_SLOT 0xFFFFu
#define SPLASH_KVZIP_SPILL_START 4u

// Group tiers as their two flag bits.
#define SPLASH_KVZIP_TIER_WINDOW8 0u
#define SPLASH_KVZIP_TIER_WINDOW16 1u
#define SPLASH_KVZIP_TIER_RAW 3u
#define SPLASH_KVZIP_TIER_LOST 2u

// The codec buffer: a header, then per attention layer and tensor (keys,
// values) and KV head, base3[256] and base4[256].
struct SplashKvZipHeader {
  // Slabs that lost groups to an exhausted spill pool, counted by the store
  // kernels.
  uint32_t overflow_slabs;
  uint32_t layers;
  uint32_t kv_heads;
  uint32_t reserved[13];
};

#define SPLASH_KVZIP_HEADER_BYTES 64u
#define SPLASH_KVZIP_BASE_BYTES_PER_HEAD (2u * SPLASH_KVZIP_DIMENSIONS)

#ifndef __METAL_VERSION__
static_assert(sizeof(SplashKvZipHeader) == SPLASH_KVZIP_HEADER_BYTES,
              "the codec header is 64 bytes on both sides");
#endif

// Where one head's bases begin in the codec buffer, in bytes.
inline uint32_t splash_kvzip_base_offset(uint32_t kv_heads, uint32_t layer,
                                         uint32_t value_tensor, uint32_t head) {
  return SPLASH_KVZIP_HEADER_BYTES +
         ((layer * 2u + value_tensor) * kv_heads + head) * SPLASH_KVZIP_BASE_BYTES_PER_HEAD;
}

inline uint32_t splash_kvzip_codec_bytes(uint32_t layers, uint32_t kv_heads) {
  return SPLASH_KVZIP_HEADER_BYTES + layers * 2u * kv_heads * SPLASH_KVZIP_BASE_BYTES_PER_HEAD;
}

// Overflow bytes the groups of a 2-bit-per-group flag word take: one per
// window-16 group, five per raw group, none for lost groups.
inline uint32_t splash_kvzip_flag_cost(uint32_t flags) {
  const uint32_t low = flags & 0x55555555u;
  const uint32_t raw = low & (flags >> 1);
#ifdef __METAL_VERSION__
  return metal::popcount(low) + 4u * metal::popcount(raw);
#else
  return uint32_t(__builtin_popcount(low)) + 4u * uint32_t(__builtin_popcount(raw));
#endif
}

// The bytes of a page's spill pool: one share per slab of every layer.
inline uint32_t splash_kvzip_spill_bytes(uint32_t layers, uint32_t kv_heads) {
  return layers * 2u * kv_heads * SPLASH_KVZIP_SHARE_BYTES;
}

// Prefill attends ZBF16 history through the BF16 kernels: before a chunk's
// attention, its committed history is expanded into a BF16 scratch of one
// layer, whose pages the scratch table names in logical order. chunk is the
// store's parameters in the ZBF16 pool; scratch places the layer's region in
// the scratch's single extent.
struct SplashKvZipExpandParams {
  SplashChunkedPrefillParams chunk;
  SplashKvLayer scratch;
};

#ifndef __METAL_VERSION__
static_assert(sizeof(SplashKvZipExpandParams) == 32,
              "ZBF16 expand parameters are 32 bytes on both sides");
#endif

