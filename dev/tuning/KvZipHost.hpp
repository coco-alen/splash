#pragma once

#include "metal/abi/KvExtent.h"
#include "metal/abi/KvZip.h"
#include "tuning/HostKvExtents.hpp"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <vector>

namespace splash::ops::tuning::kvzip {

constexpr uint32_t kRows = SPLASH_KVZIP_ROWS;
constexpr uint32_t kDims = SPLASH_KVZIP_DIMENSIONS;

// ---------------------------------------------------------------------------
// Host reference codec of ZBF16 pages (metal/abi/KvZip.h): what the store
// kernels write and the attention kernels read, for kernel tests and the
// attention fixture.

struct HeadBases final {
  std::array<uint8_t, kDims> base3{};
  std::array<uint8_t, kDims> base4{};
};

inline uint32_t flagCost(uint32_t flags) { return splash_kvzip_flag_cost(flags); }

// One ZBF16 page of a host pool: its slabs in one layer and its spill pool
// across every layer.
struct ZipPage final {
  const HostKvExtents &extents;
  uint32_t layers, kvHeads, page, layer;

  uint8_t *data(uint32_t tensor, uint32_t head) const {
    return extents.slab<uint8_t>(layer, tensor ? SPLASH_KV_VALUES : SPLASH_KV_KEYS, page) +
           head * SPLASH_KVZIP_DATA_BYTES_PER_HEAD;
  }
  uint8_t *aux(uint32_t tensor, uint32_t head) const {
    return extents.slab<uint8_t>(layer, tensor ? SPLASH_KV_VALUE_SCALES : SPLASH_KV_KEY_SCALES, page) +
           head * SPLASH_KVZIP_AUX_BYTES_PER_HEAD;
  }
  uint8_t *spill(uint32_t at) const {
    const uint32_t share = at / SPLASH_KVZIP_SHARE_BYTES;
    const uint32_t owner = share / (2 * kvHeads), tensor = share / kvHeads % 2, head = share % kvHeads;
    return extents.slab<uint8_t>(owner, tensor ? SPLASH_KV_VALUE_SCALES : SPLASH_KV_KEY_SCALES, page) +
           head * SPLASH_KVZIP_AUX_BYTES_PER_HEAD + SPLASH_KVZIP_SHARE_OFFSET + at % SPLASH_KVZIP_SHARE_BYTES;
  }
  uint32_t capacity() const { return splash_kvzip_spill_bytes(layers, kvHeads); }
  uint32_t counter() const {
    uint32_t value;
    std::memcpy(&value, spill(0), 4);
    return value;
  }
  void setCounter(uint32_t value) const { std::memcpy(spill(0), &value, 4); }
};

// One row's groups: tiers, code words and overflow bytes.
struct RowCode final {
  std::array<uint32_t, SPLASH_KVZIP_GROUPS_PER_ROW> tier{}, word{};
  std::array<std::array<uint8_t, 5>, SPLASH_KVZIP_GROUPS_PER_ROW> bytes{};
  uint32_t total = 0;
};

inline RowCode encodeRow(const uint16_t *row, const HeadBases &bases) {
  RowCode code;
  for (uint32_t group = 0; group < SPLASH_KVZIP_GROUPS_PER_ROW; ++group) {
    bool all3 = true, all4 = true;
    std::array<uint32_t, 8> exponent{};
    for (uint32_t lane = 0; lane < 8; ++lane) {
      const uint32_t dimension = group * 8 + lane;
      exponent[lane] = (row[dimension] >> 7) & 0xFF;
      const int d3 = int(exponent[lane]) - bases.base3[dimension];
      const int d4 = int(exponent[lane]) - bases.base4[dimension];
      all3 &= d3 >= 0 && d3 < 8;
      all4 &= d4 >= 0 && d4 < 16;
    }
    const uint32_t tier = all3 ? 0 : all4 ? 1 : 3;
    uint32_t word = 0, window16 = 0;
    uint64_t high = 0;
    for (uint32_t lane = 0; lane < 8; ++lane) {
      const uint32_t dimension = group * 8 + lane;
      const uint32_t value = tier == 0   ? exponent[lane] - bases.base3[dimension]
                             : tier == 1 ? exponent[lane] - bases.base4[dimension]
                                         : exponent[lane];
      word |= (value & 7) << (3 * lane);
      window16 |= ((value >> 3) & 1) << lane;
      high |= uint64_t(exponent[lane] >> 3) << (5 * lane);
    }
    code.tier[group] = tier;
    code.word[group] = word;
    for (uint32_t byte = 0; byte < 5; ++byte)
      code.bytes[group][byte] = tier == 1 ? (byte ? 0 : uint8_t(window16)) : uint8_t(high >> (8 * byte));
    code.total += flagCost(tier);
  }
  return code;
}

// Stores rows [begin, end) of a slab as the store kernels do; `rows` holds
// the page's 32 token-major rows. Returns whether a row lost groups.
inline bool storeRows(const ZipPage &page, uint32_t tensor, uint32_t head, const uint16_t *rows, uint32_t begin,
               uint32_t end, const HeadBases &bases) {
  uint8_t *data = page.data(tensor, head), *aux = page.aux(tensor, head);
  auto *flags = reinterpret_cast<uint32_t *>(aux);
  auto *spill = reinterpret_cast<uint16_t *>(aux + SPLASH_KVZIP_SPILL_ROWS_OFFSET);
  uint8_t *slot = aux + SPLASH_KVZIP_SLOT_OFFSET;
  uint32_t used = 0;
  for (uint32_t row = 0; row < begin; ++row)
    if (spill[row] == SPLASH_KVZIP_IN_SLOT) used += flagCost(flags[2 * row]) + flagCost(flags[2 * row + 1]);
  bool lostAny = false;
  for (uint32_t row = begin; row < end; ++row) {
    RowCode code = encodeRow(rows + row * kDims, bases);
    uint32_t at = SPLASH_KVZIP_IN_SLOT;
    bool lost = false;
    if (code.total && used + code.total > SPLASH_KVZIP_SLOT_BYTES) {
      at = page.counter();
      page.setCounter(at + code.total);
      lost = at + code.total > page.capacity();
    }
    lostAny |= lost;
    std::array<uint32_t, 2> words{};
    uint32_t offset = 0;
    for (uint32_t group = 0; group < SPLASH_KVZIP_GROUPS_PER_ROW; ++group) {
      uint32_t tier = code.tier[group];
      if (lost && tier) tier = SPLASH_KVZIP_TIER_LOST;
      const uint32_t word = tier == SPLASH_KVZIP_TIER_LOST ? 0 : code.word[group];
      uint8_t *codes = data + SPLASH_KVZIP_SM_BYTES + row * SPLASH_KVZIP_CODE_BYTES_PER_ROW + group * 3;
      codes[0] = uint8_t(word), codes[1] = uint8_t(word >> 8), codes[2] = uint8_t(word >> 16);
      words[group / 16] |= tier << (2 * (group % 16));
      for (uint32_t byte = 0; byte < flagCost(tier); ++byte) {
        if (at == SPLASH_KVZIP_IN_SLOT) slot[used + offset + byte] = code.bytes[group][byte];
        else *page.spill(at + offset + byte) = code.bytes[group][byte];
      }
      offset += flagCost(tier);
    }
    for (uint32_t dimension = 0; dimension < kDims; ++dimension) {
      const uint32_t bits = rows[row * kDims + dimension];
      data[row * kDims + dimension] = uint8_t(((bits >> 8) & 0x80) | (bits & 0x7F));
    }
    flags[2 * row] = words[0], flags[2 * row + 1] = words[1];
    spill[row] = uint16_t(lost ? SPLASH_KVZIP_IN_SLOT : at);
    if (at == SPLASH_KVZIP_IN_SLOT) used += code.total;
  }
  return lostAny;
}

// Decodes rows [0, rows) of a slab into BF16 bits, token-major; a lost group
// decodes as the kernels decode it.
inline std::vector<uint16_t> decodeSlab(const ZipPage &page, uint32_t tensor, uint32_t head, uint32_t rows,
                                 const HeadBases &bases) {
  const uint8_t *data = page.data(tensor, head), *aux = page.aux(tensor, head);
  const auto *flags = reinterpret_cast<const uint32_t *>(aux);
  const auto *spill = reinterpret_cast<const uint16_t *>(aux + SPLASH_KVZIP_SPILL_ROWS_OFFSET);
  const uint8_t *slot = aux + SPLASH_KVZIP_SLOT_OFFSET;
  std::vector<uint16_t> result(rows * kDims);
  uint32_t used = 0;
  for (uint32_t row = 0; row < rows; ++row) {
    const bool inSlot = spill[row] == SPLASH_KVZIP_IN_SLOT;
    uint32_t offset = inSlot ? used : spill[row];
    for (uint32_t group = 0; group < SPLASH_KVZIP_GROUPS_PER_ROW; ++group) {
      const uint32_t tier = (flags[2 * row + group / 16] >> (2 * (group % 16))) & 3;
      const uint8_t *codes = data + SPLASH_KVZIP_SM_BYTES + row * SPLASH_KVZIP_CODE_BYTES_PER_ROW + group * 3;
      const uint32_t word = codes[0] | codes[1] << 8 | codes[2] << 16;
      uint64_t high = 0;
      for (uint32_t byte = 0; byte < flagCost(tier); ++byte)
        high |= uint64_t(inSlot ? slot[offset + byte] : *page.spill(offset + byte)) << (8 * byte);
      offset += flagCost(tier);
      for (uint32_t lane = 0; lane < 8; ++lane) {
        const uint32_t dimension = group * 8 + lane;
        const uint32_t value = (word >> (3 * lane)) & 7;
        const uint32_t exponent = tier == 0   ? bases.base3[dimension] + value
                                  : tier == 1 ? bases.base4[dimension] + (value | ((high >> lane) & 1) << 3)
                                  : tier == 3 ? value | uint32_t((high >> (5 * lane)) & 31) << 3
                                              : std::min<uint32_t>(bases.base3[dimension] + value, 254);
        const uint32_t sm = data[row * kDims + dimension];
        result[row * kDims + dimension] =
            uint16_t(((sm & 0x80) << 8) | ((exponent & 0xFF) << 7) | (sm & 0x7F));
      }
    }
    if (inSlot) used = offset;
  }
  return result;
}


// The best 8- and 16-binade windows of one (tensor, KV head) from each
// dimension's exponent histogram, as dev/tools/kvzip_calibrate.py fits them.
inline HeadBases windows(const std::vector<std::array<uint32_t, 256>> &histograms) {
  HeadBases bases;
  for (uint32_t dimension = 0; dimension < kDims; ++dimension) {
    std::array<uint64_t, 257> cumulative{};
    for (uint32_t exponent = 0; exponent < 256; ++exponent)
      cumulative[exponent + 1] = cumulative[exponent] + histograms[dimension][exponent];
    for (uint32_t width : {8U, 16U}) {
      uint64_t best = 0;
      uint32_t base = 0;
      for (uint32_t start = 0; start + width <= 256; ++start)
        if (cumulative[start + width] - cumulative[start] > best)
          best = cumulative[start + width] - cumulative[start], base = start;
      (width == 8 ? bases.base3 : bases.base4)[dimension] = uint8_t(base);
    }
  }
  return bases;
}

} // namespace splash::ops::tuning::kvzip
