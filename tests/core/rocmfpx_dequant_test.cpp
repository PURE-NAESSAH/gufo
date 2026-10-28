#include "src/core/quant/ggml_dequant.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <numeric>
#include <span>
#include <string_view>
#include <vector>

#include "src/core/gguf_reader.hpp"

namespace {

void ExpectTrue(bool condition, std::string_view message) {
  if (!condition) {
    std::cerr << "Assertion failed: " << message << '\n';
    std::exit(1);
  }
}

void ExpectNear(float actual, float expected, std::string_view message,
                float tolerance = 1.0e-5F) {
  if (std::abs(actual - expected) > tolerance) {
    std::cerr << "Assertion failed: " << message << " actual=" << actual
              << " expected=" << expected << '\n';
    std::exit(1);
  }
}

void TestBlockElementsAndRowBytes() {
  using gufo::core::GgmlType;
  using gufo::quant::QuantizedBlockElements;
  using gufo::quant::QuantizedRowBytes;

  ExpectTrue(QuantizedBlockElements(GgmlType::kQ4_0_ROCMFP4) == 32, "block rocmfp4 == 32");
  ExpectTrue(QuantizedBlockElements(GgmlType::kQ4_0_ROCMFP4_FAST) == 32, "block rocmfp4_fast == 32");
  ExpectTrue(QuantizedBlockElements(GgmlType::kQ2_0_ROCMFPX) == 32, "block rocmfp2 == 32");
  ExpectTrue(QuantizedBlockElements(GgmlType::kQ3_0_ROCMFPX) == 32, "block rocmfp3 == 32");
  ExpectTrue(QuantizedBlockElements(GgmlType::kQ6_0_ROCMFPX) == 32, "block rocmfp6 == 32");
  ExpectTrue(QuantizedBlockElements(GgmlType::kQ8_0_ROCMFPX) == 32, "block rocmfp8 == 32");
  ExpectTrue(QuantizedBlockElements(GgmlType::kQ4_0_ROCMI4) == 32, "block rocmi4 == 32");

  ExpectTrue(QuantizedRowBytes(GgmlType::kQ4_0_ROCMFP4, 32) == 18, "row bytes rocmfp4 32");
  ExpectTrue(QuantizedRowBytes(GgmlType::kQ4_0_ROCMFP4_FAST, 32) == 17, "row bytes rocmfp4_fast 32");
  ExpectTrue(QuantizedRowBytes(GgmlType::kQ2_0_ROCMFPX, 32) == 10, "row bytes rocmfp2 32");
  ExpectTrue(QuantizedRowBytes(GgmlType::kQ3_0_ROCMFPX, 32) == 14, "row bytes rocmfp3 32");
  ExpectTrue(QuantizedRowBytes(GgmlType::kQ6_0_ROCMFPX, 32) == 26, "row bytes rocmfp6 32");
  ExpectTrue(QuantizedRowBytes(GgmlType::kQ8_0_ROCMFPX, 32) == 33, "row bytes rocmfp8 32");
  ExpectTrue(QuantizedRowBytes(GgmlType::kQ4_0_ROCMI4, 32) == 17, "row bytes rocmi4 32");

  ExpectTrue(QuantizedRowBytes(GgmlType::kQ4_0_ROCMFP4_FAST, 64) == 34, "row bytes rocmfp4_fast 64");
}

void TestRocmFp4Fast() {
  // Test Q4_0_ROCMFP4_FAST
  gufo::quant::block_rocmfp4_fast block{};
  // e = 0x58: exp = 11, mant = 0 -> (8 + 0) * 2^0 = 8.0f -> half-scale is 8.0f!
  block.e = 0x58;
  for (std::size_t j = 0; j < 16; ++j) {
    const std::uint8_t lo = static_cast<std::uint8_t>(j % 16);
    const std::uint8_t hi = static_cast<std::uint8_t>((15 - j) % 16);
    block.qs[j] = static_cast<std::uint8_t>(lo | (hi << 4U));
  }

  std::array<float, 32> dst{};
  gufo::quant::DequantizeQ4_0_ROCMFP4_FAST(&block, dst.data(), 32);

  constexpr std::int8_t codebook[16] = {
      0, 1, 2, 3, 4, 6, 8, 10,
      0, -1, -2, -3, -4, -6, -8, -10,
  };

  const float expected_scale = 8.0f;
  for (std::size_t j = 0; j < 16; ++j) {
    const float exp_lo = static_cast<float>(codebook[j % 16]) * expected_scale;
    const float exp_hi = static_cast<float>(codebook[(15 - j) % 16]) * expected_scale;
    ExpectNear(dst[j], exp_lo, "rocmfp4_fast dequant lo");
    ExpectNear(dst[j + 16], exp_hi, "rocmfp4_fast dequant hi");
  }

  std::vector<float> vec(32, 1.0f);
  const float dot = gufo::quant::DotProductQ4_0_ROCMFP4_FAST(&block, vec, 32);
  const float expected_dot = std::accumulate(dst.begin(), dst.end(), 0.0f);
  ExpectNear(dot, expected_dot, "rocmfp4_fast dot product");
}

void TestRocmFp4DualScale() {
  gufo::quant::block_rocmfp4 block{};
  // Half 0: e[0] = 0x58 (scale = 8.0f)
  // Half 1: e[1] = 0x50 (scale = 4.0f)
  block.e[0] = 0x58;
  block.e[1] = 0x50;
  for (std::size_t j = 0; j < 16; ++j) {
    const std::uint8_t lo = static_cast<std::uint8_t>(j % 16);
    const std::uint8_t hi = static_cast<std::uint8_t>((15 - j) % 16);
    block.qs[j] = static_cast<std::uint8_t>(lo | (hi << 4U));
  }

  std::array<float, 32> dst{};
  gufo::quant::DequantizeQ4_0_ROCMFP4(&block, dst.data(), 32);

  constexpr std::int8_t codebook[16] = {
      0, 1, 2, 3, 4, 6, 8, 10,
      0, -1, -2, -3, -4, -6, -8, -10,
  };

  for (std::size_t j = 0; j < 16; ++j) {
    const float exp_lo = static_cast<float>(codebook[j % 16]) * 8.0f;
    const float exp_hi = static_cast<float>(codebook[(15 - j) % 16]) * 4.0f;
    ExpectNear(dst[j], exp_lo, "rocmfp4 dual scale dequant lo");
    ExpectNear(dst[j + 16], exp_hi, "rocmfp4 dual scale dequant hi");
  }

  std::vector<float> vec(32);
  for (std::size_t i = 0; i < 32; ++i) {
    vec[i] = static_cast<float>(i + 1);
  }
  const float dot = gufo::quant::DotProductQ4_0_ROCMFP4(&block, vec, 32);
  float expected_dot = 0.0f;
  for (std::size_t i = 0; i < 32; ++i) {
    expected_dot += dst[i] * vec[i];
  }
  ExpectNear(dot, expected_dot, "rocmfp4 dual scale dot product");
}

void TestRocmFp2() {
  gufo::quant::block_rocmfp2 block{};
  block.e[0] = 0x58; // scale 8.0
  block.e[1] = 0x58;
  // FP2 codes: 0 -> -4, 1 -> -1, 2 -> 1, 3 -> 4
  // pack 4 codes per byte
  for (std::size_t b = 0; b < 8; ++b) {
    block.qs[b] = 0b11'10'01'00; // 0, 1, 2, 3
  }
  std::array<float, 32> dst{};
  gufo::quant::DequantizeQ2_0_ROCMFPX(&block, dst.data(), 32);
  for (std::size_t i = 0; i < 32; ++i) {
    float exp = 0.0f;
    switch (i % 4) {
      case 0: exp = -4.0f * 8.0f; break;
      case 1: exp = -1.0f * 8.0f; break;
      case 2: exp =  1.0f * 8.0f; break;
      case 3: exp =  4.0f * 8.0f; break;
    }
    ExpectNear(dst[i], exp, "rocmfp2 dequant");
  }

  std::vector<float> vec(32, 0.5f);
  const float dot = gufo::quant::DotProductQ2_0_ROCMFPX(&block, vec, 32);
  const float expected_dot = std::accumulate(dst.begin(), dst.end(), 0.0f) * 0.5f;
  ExpectNear(dot, expected_dot, "rocmfp2 dot product");
}

void TestRocmI4() {
  gufo::quant::block_rocmi4 block{};
  block.e = 0x58; // scale 8.0
  for (std::size_t j = 0; j < 16; ++j) {
    // lo = j (0..15), hi = 15 - j
    block.qs[j] = static_cast<std::uint8_t>((j & 0x0F) | (((15 - j) & 0x0F) << 4U));
  }
  std::array<float, 32> dst{};
  gufo::quant::DequantizeQ4_0_ROCMI4(&block, dst.data(), 32);

  // 4-bit two's complement: 0..7 are positive, 8..15 are -8..-1
  const auto decode_i4 = [](std::uint8_t n) -> float {
    return static_cast<float>(static_cast<std::int8_t>((n & 0x08) ? static_cast<int>(n | 0xF0) : static_cast<int>(n & 0x07)));
  };

  for (std::size_t j = 0; j < 16; ++j) {
    ExpectNear(dst[j], decode_i4(static_cast<std::uint8_t>(j)) * 8.0f, "rocmi4 dequant lo");
    ExpectNear(dst[j + 16], decode_i4(static_cast<std::uint8_t>(15 - j)) * 8.0f, "rocmi4 dequant hi");
  }

  std::vector<float> vec(32, 1.0f);
  const float dot = gufo::quant::DotProductQ4_0_ROCMI4(&block, vec, 32);
  const float expected_dot = std::accumulate(dst.begin(), dst.end(), 0.0f);
  ExpectNear(dot, expected_dot, "rocmi4 dot product");
}

void TestRocmFp8() {
  gufo::quant::block_rocmfp8 block{};
  block.e = 0x58; // scale 8.0
  for (std::size_t i = 0; i < 32; ++i) {
    block.qs[i] = static_cast<std::int8_t>(static_cast<int>(i) - 16);
  }
  std::array<float, 32> dst{};
  gufo::quant::DequantizeQ8_0_ROCMFPX(&block, dst.data(), 32);
  for (std::size_t i = 0; i < 32; ++i) {
    ExpectNear(dst[i], static_cast<float>(static_cast<int>(i) - 16) * 8.0f, "rocmfp8 dequant");
  }
  std::vector<float> vec(32, 1.0f);
  const float dot = gufo::quant::DotProductQ8_0_ROCMFPX(&block, vec, 32);
  const float expected_dot = std::accumulate(dst.begin(), dst.end(), 0.0f);
  ExpectNear(dot, expected_dot, "rocmfp8 dot product");
}

void TestRocmFp3() {
  gufo::quant::block_rocmfp3 block{};
  block.e[0] = 0x58; // scale 8.0
  block.e[1] = 0x58;

  const auto pack8 = [](std::uint8_t* dst, const std::uint8_t* c) {
    dst[0] = static_cast<std::uint8_t>((c[0] & 7U) | ((c[1] & 7U) << 3U) | ((c[2] & 3U) << 6U));
    dst[1] = static_cast<std::uint8_t>(((c[2] >> 2U) & 1U) | ((c[3] & 7U) << 1U) | ((c[4] & 7U) << 4U) | ((c[5] & 1U) << 7U));
    dst[2] = static_cast<std::uint8_t>(((c[5] >> 1U) & 3U) | ((c[6] & 7U) << 2U) | ((c[7] & 7U) << 5U));
  };

  std::array<std::uint8_t, 32> codes{};
  for (std::size_t i = 0; i < 32; ++i) {
    codes[i] = static_cast<std::uint8_t>(i % 8);
  }
  pack8(block.qs,      codes.data());
  pack8(block.qs + 3,  codes.data() + 8);
  pack8(block.qs + 6,  codes.data() + 16);
  pack8(block.qs + 9,  codes.data() + 24);

  std::array<float, 32> dst{};
  gufo::quant::DequantizeQ3_0_ROCMFPX(&block, dst.data(), 32);

  constexpr int mag[4] = {0, 1, 2, 4};
  for (std::size_t i = 0; i < 32; ++i) {
    const int c = codes[i];
    const int val = mag[c & 3];
    const float exp = static_cast<float>((c & 4) ? -val : val) * 8.0f;
    ExpectNear(dst[i], exp, "rocmfp3 dequant");
  }

  std::vector<float> vec(32, 1.0f);
  const float dot = gufo::quant::DotProductQ3_0_ROCMFPX(&block, vec, 32);
  const float expected_dot = std::accumulate(dst.begin(), dst.end(), 0.0f);
  ExpectNear(dot, expected_dot, "rocmfp3 dot product");
}

void TestRocmFp6() {
  gufo::quant::block_rocmfp6 block{};
  block.e[0] = 0x58; // scale 8.0
  block.e[1] = 0x58;

  const auto pack4 = [](std::uint8_t* dst, const std::uint8_t* c) {
    dst[0] = static_cast<std::uint8_t>((c[0] & 0x3FU) | ((c[1] & 0x03U) << 6U));
    dst[1] = static_cast<std::uint8_t>(((c[1] >> 2U) & 0x0FU) | ((c[2] & 0x0FU) << 4U));
    dst[2] = static_cast<std::uint8_t>(((c[2] >> 4U) & 0x03U) | ((c[3] & 0x3FU) << 2U));
  };

  std::array<std::uint8_t, 32> codes{};
  for (std::size_t i = 0; i < 32; ++i) {
    codes[i] = static_cast<std::uint8_t>((i * 2) % 64);
  }
  for (std::size_t g = 0; g < 8; ++g) {
    pack4(block.qs + (g * 3), codes.data() + (g * 4));
  }

  std::array<float, 32> dst{};
  gufo::quant::DequantizeQ6_0_ROCMFPX(&block, dst.data(), 32);

  for (std::size_t i = 0; i < 32; ++i) {
    const int c = codes[i];
    const int m = c & 31;
    const int val = (c & 32) ? -(m == 0 ? 32 : m) : m;
    const float exp = static_cast<float>(val) * 8.0f;
    ExpectNear(dst[i], exp, "rocmfp6 dequant");
  }

  std::vector<float> vec(32, 1.0f);
  const float dot = gufo::quant::DotProductQ6_0_ROCMFPX(&block, vec, 32);
  const float expected_dot = std::accumulate(dst.begin(), dst.end(), 0.0f);
  ExpectNear(dot, expected_dot, "rocmfp6 dot product");
}

}  // namespace

int main() {
  TestBlockElementsAndRowBytes();
  TestRocmFp4Fast();
  TestRocmFp4DualScale();
  TestRocmFp2();
  TestRocmFp3();
  TestRocmFp6();
  TestRocmI4();
  TestRocmFp8();
  std::cout << "All ROCmFPX dequantization tests passed successfully.\n";
  return 0;
}
