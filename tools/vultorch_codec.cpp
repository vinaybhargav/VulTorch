#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <cmath>
#include <cstring>
#include <chrono>
#include <algorithm>
#include <cstdint>
#include <immintrin.h>

#ifdef _OPENMP
#include <omp.h>
#endif

// ============================================================================
// 1. Model Structures & Zero-Copy Binary Loader
// ============================================================================

struct QLinear {
    uint32_t in_features;
    uint32_t out_features;
    float scale;
    std::vector<int8_t> qweight; // [out_features * in_features]
    std::vector<float> bias;     // [out_features]

    void forward(const float* in, float* out) const {
        const float s = scale;
        for (uint32_t m = 0; m < out_features; ++m) {
            const int8_t* w_row = &qweight[m * in_features];
            float sum = 0.0f;

            uint32_t k = 0;
#if defined(__AVX2__)
            __m256 vsum = _mm256_setzero_ps();
            for (; k + 8 <= in_features; k += 8) {
                __m256 vin = _mm256_loadu_ps(&in[k]);
                // Load 8 x int8 and convert to float
                __m128i vi8 = _mm_loadu_si64(&w_row[k]);
                __m256i vi32 = _mm256_cvtepi8_epi32(vi8);
                __m256 vw = _mm256_cvtepi32_ps(vi32);
                vsum = _mm256_fmadd_ps(vin, vw, vsum);
            }
            // Horizontal sum
            alignas(32) float tmp[8];
            _mm256_store_ps(tmp, vsum);
            for (int i = 0; i < 8; ++i) sum += tmp[i];
#endif
            for (; k < in_features; ++k) {
                sum += in[k] * static_cast<float>(w_row[k]);
            }
            out[m] = sum * s + bias[m];
        }
    }
};

struct LayerNorm {
    uint32_t dim;
    std::vector<float> weight;
    std::vector<float> bias;

    void forward(float* x) const {
        float mean = 0.0f;
        for (uint32_t i = 0; i < dim; ++i) mean += x[i];
        mean /= dim;

        float var = 0.0f;
        for (uint32_t i = 0; i < dim; ++i) {
            float d = x[i] - mean;
            var += d * d;
        }
        var /= dim;
        float inv_std = 1.0f / std::sqrt(var + 1e-5f);

        for (uint32_t i = 0; i < dim; ++i) {
            x[i] = ((x[i] - mean) * inv_std) * weight[i] + bias[i];
        }
    }
};

static inline float gelu_act(float x) {
    return 0.5f * x * (1.0f + std::tanh(0.7978845608f * (x + 0.044715f * x * x * x)));
}

class NeuralCompressorEngine {
public:
    // Forward() uses fixed-size stack buffers; these are their capacities and
    // the contract the loader validates against.
    static constexpr uint32_t kMaxContextLen = 256;
    static constexpr uint32_t kMaxEmbedDim = 512;
    static constexpr uint32_t kMaxHiddenDim = 1024;

    uint32_t context_len = 32;
    uint32_t embed_dim = 512;
    uint32_t hidden_dim = 1024;
    uint32_t vocab_size = 256;

    std::vector<float> embed_weight; // [256 * 512]
    QLinear fc1;
    LayerNorm ln1;
    QLinear fc2;
    LayerNorm ln2;
    QLinear fc3;
    LayerNorm ln3;
    QLinear fc_out;

    bool load(const std::string& path) {
        std::ifstream file(path, std::ios::binary);
        if (!file.is_open()) return false;

        char magic[4];
        file.read(magic, 4);
        if (std::memcmp(magic, "VTC1", 4) != 0) return false;

        // The container is big-endian regardless of host byte order.
        auto read_u32 = [&]() {
            uint32_t val = 0;
            file.read(reinterpret_cast<char*>(&val), 4);
            return __builtin_bswap32(val);
        };
        auto read_f32 = [&]() {
            uint32_t val = 0;
            file.read(reinterpret_cast<char*>(&val), 4);
            val = __builtin_bswap32(val);
            float f = 0.0f;
            std::memcpy(&f, &val, 4);
            return f;
        };

        const uint32_t version = read_u32();
        const uint32_t ctx_l = read_u32();
        const uint32_t emb_d = read_u32();
        const uint32_t hid_d = read_u32();
        const uint32_t voc_s = read_u32();
        const uint32_t declared_params = read_u32();

        // Validate before trusting: forward() writes into fixed-size buffers,
        // so a mismatched header would silently overflow the stack.
        if (version != 1) {
            std::cerr << "Unsupported model version: " << version << "\n";
            return false;
        }
        if (ctx_l == 0 || ctx_l > kMaxContextLen ||
            emb_d == 0 || emb_d > kMaxEmbedDim ||
            hid_d == 0 || hid_d > kMaxHiddenDim) {
            std::cerr << "Model dimensions out of supported range: ctx=" << ctx_l
                      << " emb=" << emb_d << " hid=" << hid_d << "\n";
            return false;
        }
        if (voc_s != 256) {
            std::cerr << "Only a 256-symbol byte vocabulary is supported, got " << voc_s << "\n";
            return false;
        }
        if (declared_params == 0) {
            std::cerr << "Model header declares zero parameters\n";
            return false;
        }

        context_len = ctx_l;
        embed_dim = emb_d;
        hidden_dim = hid_d;
        vocab_size = voc_s;

        // 1. Embeddings
        embed_weight.resize(vocab_size * embed_dim);
        file.read(reinterpret_cast<char*>(embed_weight.data()), embed_weight.size() * sizeof(float));

        auto load_qlinear = [&](QLinear& ql, uint32_t in_f, uint32_t out_f) {
            ql.in_features = in_f;
            ql.out_features = out_f;
            ql.scale = read_f32();
            ql.qweight.resize(in_f * out_f);
            file.read(reinterpret_cast<char*>(ql.qweight.data()), ql.qweight.size());
            ql.bias.resize(out_f);
            file.read(reinterpret_cast<char*>(ql.bias.data()), out_f * sizeof(float));
        };

        auto load_ln = [&](LayerNorm& ln, uint32_t d) {
            ln.dim = d;
            ln.weight.resize(d);
            ln.bias.resize(d);
            file.read(reinterpret_cast<char*>(ln.weight.data()), d * sizeof(float));
            file.read(reinterpret_cast<char*>(ln.bias.data()), d * sizeof(float));
        };

        // 2. FC1 & LN1
        load_qlinear(fc1, context_len * embed_dim, 512);
        load_ln(ln1, 512);

        // 3. FC2 & LN2
        load_qlinear(fc2, 512, hidden_dim);
        load_ln(ln2, hidden_dim);

        // 4. FC3 & LN3
        load_qlinear(fc3, hidden_dim, 512);
        load_ln(ln3, 512);

        // 5. FC_OUT
        load_qlinear(fc_out, 512, 256);

        return true;
    }

    void forward(const uint8_t* ctx, float* probs) const {
        // Flattened embedding buffer
        alignas(32) std::vector<float> flat(context_len * embed_dim);
        for (uint32_t i = 0; i < context_len; ++i) {
            const float* emb_row = &embed_weight[ctx[i] * embed_dim];
            std::memcpy(&flat[i * embed_dim], emb_row, embed_dim * sizeof(float));
        }

        // Layer 1
        alignas(32) float h1[kMaxEmbedDim];
        fc1.forward(flat.data(), h1);
        ln1.forward(h1);
        for (uint32_t i = 0; i < fc1.out_features; ++i) h1[i] = gelu_act(h1[i]);

        // Layer 2
        alignas(32) float h2[kMaxHiddenDim];
        fc2.forward(h1, h2);
        ln2.forward(h2);
        for (uint32_t i = 0; i < fc2.out_features; ++i) h2[i] = gelu_act(h2[i]);

        // Layer 3
        alignas(32) float h3[kMaxEmbedDim];
        fc3.forward(h2, h3);
        ln3.forward(h3);
        for (uint32_t i = 0; i < fc3.out_features; ++i) h3[i] = gelu_act(h3[i]);

        // Output logits
        alignas(32) float logits[256];
        fc_out.forward(h3, logits);

        // Softmax
        float max_l = logits[0];
        for (uint32_t i = 1; i < vocab_size; ++i) if (logits[i] > max_l) max_l = logits[i];

        float sum_exp = 0.0f;
        for (uint32_t i = 0; i < vocab_size; ++i) {
            probs[i] = std::exp(logits[i] - max_l);
            sum_exp += probs[i];
        }
        float inv_sum = 1.0f / sum_exp;
        for (uint32_t i = 0; i < vocab_size; ++i) probs[i] *= inv_sum;
    }
};

// ============================================================================
// 2. High-Precision 32-bit Integer Range Coder
// ============================================================================

class RangeEncoder {
public:
    static constexpr uint32_t TOP = 1 << 24;
    static constexpr uint32_t BOTTOM = 1 << 16;
    static constexpr uint32_t TOTAL_COUNT = 1 << 14;

    uint32_t low = 0;
    uint32_t range = 0xFFFFFFFF;
    std::vector<uint8_t> output;

    void encode(uint8_t sym, const float* probs) {
        // Build cumulative table
        uint32_t counts[256];
        uint32_t current_sum = 0;
        for (int i = 0; i < 256; ++i) {
            counts[i] = std::max(1, static_cast<int>(probs[i] * (TOTAL_COUNT - 256)));
            current_sum += counts[i];
        }
        counts[0] += (TOTAL_COUNT - current_sum);

        uint32_t cum[257];
        cum[0] = 0;
        for (int i = 0; i < 256; ++i) cum[i + 1] = cum[i] + counts[i];

        uint32_t sym_low = cum[sym];
        uint32_t sym_high = cum[sym + 1];

        range /= TOTAL_COUNT;
        low += sym_low * range;
        range *= (sym_high - sym_low);

        while ((low ^ (low + range)) < TOP || (range < BOTTOM && ((range = -low & (BOTTOM - 1)), true))) {
            output.push_back(static_cast<uint8_t>((low >> 24) & 0xFF));
            low = (low << 8) & 0xFFFFFFFF;
            range = (range << 8) & 0xFFFFFFFF;
        }
    }

    std::vector<uint8_t> finish() {
        for (int i = 0; i < 4; ++i) {
            output.push_back(static_cast<uint8_t>((low >> 24) & 0xFF));
            low = (low << 8) & 0xFFFFFFFF;
        }
        return output;
    }
};

class RangeDecoder {
public:
    static constexpr uint32_t TOP = 1 << 24;
    static constexpr uint32_t BOTTOM = 1 << 16;
    static constexpr uint32_t TOTAL_COUNT = 1 << 14;

    uint32_t low = 0;
    uint32_t range = 0xFFFFFFFF;
    uint32_t code = 0;
    size_t stream_idx = 0;
    const uint8_t* stream = nullptr;
    size_t stream_len = 0;

    void init(const uint8_t* data, size_t len) {
        stream = data;
        stream_len = len;
        stream_idx = 0;
        code = 0;
        for (int i = 0; i < 4; ++i) {
            uint8_t b = (stream_idx < stream_len) ? stream[stream_idx++] : 0;
            code = (code << 8) | b;
        }
    }

    uint8_t decode(const float* probs) {
        uint32_t counts[256];
        uint32_t current_sum = 0;
        for (int i = 0; i < 256; ++i) {
            counts[i] = std::max(1, static_cast<int>(probs[i] * (TOTAL_COUNT - 256)));
            current_sum += counts[i];
        }
        counts[0] += (TOTAL_COUNT - current_sum);

        uint32_t cum[257];
        cum[0] = 0;
        for (int i = 0; i < 256; ++i) cum[i + 1] = cum[i] + counts[i];

        range /= TOTAL_COUNT;
        uint32_t count_val = (code - low) / range;

        // Binary search for symbol
        auto it = std::upper_bound(cum, cum + 257, count_val);
        uint8_t sym = static_cast<uint8_t>((it - cum) - 1);

        uint32_t sym_low = cum[sym];
        uint32_t sym_high = cum[sym + 1];

        low += sym_low * range;
        range *= (sym_high - sym_low);

        while ((low ^ (low + range)) < TOP || (range < BOTTOM && ((range = -low & (BOTTOM - 1)), true))) {
            uint8_t b = (stream_idx < stream_len) ? stream[stream_idx++] : 0;
            code = ((code << 8) | b) & 0xFFFFFFFF;
            low = (low << 8) & 0xFFFFFFFF;
            range = (range << 8) & 0xFFFFFFFF;
        }

        return sym;
    }
};

// ============================================================================
// 3. CLI Command Dispatcher
// ============================================================================

void print_usage(const char* prog) {
    std::cout << "VulTorch Universal Neural Text Codec (C++ Release v1.0)\n"
              << "High-Throughput AVX2/OpenMP Neural Range Compression Engine\n\n"
              << "Usage:\n"
              << "  " << prog << " -c <model.bin> <input_file> <output.vtc>     (Compress)\n"
              << "  " << prog << " -d <model.bin> <input.vtc> <restored_file>   (Decompress)\n"
              << "  " << prog << " -t <model.bin> <input_file>                  (Benchmark & Verify)\n";
}

int main(int argc, char* argv[]) {
    if (argc < 4) {
        print_usage(argv[0]);
        return 1;
    }

    std::string mode = argv[1];
    std::string model_path = argv[2];

    NeuralCompressorEngine engine;
    auto t0_load = std::chrono::high_resolution_clock::now();
    if (!engine.load(model_path)) {
        std::cerr << "❌ Error: Failed to load model weights from " << model_path << "\n";
        return 1;
    }
    auto t1_load = std::chrono::high_resolution_clock::now();
    double load_ms = std::chrono::duration<double, std::milli>(t1_load - t0_load).count();
    std::cout << "✅ Loaded 10M Q8 Model in " << load_ms << " ms\n";

    if (mode == "-c") {
        std::string in_path = argv[3];
        std::string out_path = argv[4];

        std::ifstream in_file(in_path, std::ios::binary);
        std::vector<uint8_t> data((std::istreambuf_iterator<char>(in_file)), std::istreambuf_iterator<char>());
        size_t N = data.size();

        std::cout << "📦 Compressing " << in_path << " (" << N << " bytes) with parallel batching...\n";
        auto t0_comp = std::chrono::high_resolution_clock::now();

        // 1. Parallel Batch Inference for all tokens
        const uint32_t K = engine.context_len;
        std::vector<uint8_t> padded(K, 0);
        padded.insert(padded.end(), data.begin(), data.end());

        std::vector<std::vector<float>> all_probs(N, std::vector<float>(256));

#pragma omp parallel for schedule(dynamic, 64)
        for (size_t i = 0; i < N; ++i) {
            engine.forward(&padded[i], all_probs[i].data());
        }

        // 2. Sequential Range Encode
        RangeEncoder encoder;
        for (size_t i = 0; i < N; ++i) {
            encoder.encode(data[i], all_probs[i].data());
        }
        std::vector<uint8_t> payload = encoder.finish();

        // Write output: 4-byte size header + payload
        std::ofstream out_file(out_path, std::ios::binary);
        uint32_t big_n = __builtin_bswap32(static_cast<uint32_t>(N));
        out_file.write(reinterpret_cast<const char*>(&big_n), 4);
        out_file.write(reinterpret_cast<const char*>(payload.data()), payload.size());

        auto t1_comp = std::chrono::high_resolution_clock::now();
        double comp_sec = std::chrono::duration<double>(t1_comp - t0_comp).count();
        size_t total_out = payload.size() + 4;

        double ratio = (static_cast<double>(total_out) / N) * 100.0;
        double bpc = (total_out * 8.0) / N;
        double mb_s = (N / (1024.0 * 1024.0)) / comp_sec;

        std::cout << "🎉 Compression Complete!\n"
                  << "   Compressed File: " << out_path << " (" << total_out << " bytes)\n"
                  << "   Ratio:           " << ratio << "% (" << bpc << " bits/byte)\n"
                  << "   Throughput:      " << mb_s << " MB/s (" << comp_sec << "s)\n";

    } else if (mode == "-d") {
        std::string in_path = argv[3];
        std::string out_path = argv[4];

        std::ifstream in_file(in_path, std::ios::binary);
        uint32_t big_n = 0;
        in_file.read(reinterpret_cast<char*>(&big_n), 4);
        uint32_t N = __builtin_bswap32(big_n);

        std::vector<uint8_t> payload((std::istreambuf_iterator<char>(in_file)), std::istreambuf_iterator<char>());

        std::cout << "🔄 Decompressing " << in_path << " -> " << out_path << " (" << N << " bytes)...\n";
        auto t0_decomp = std::chrono::high_resolution_clock::now();

        RangeDecoder decoder;
        decoder.init(payload.data(), payload.size());

        const uint32_t K = engine.context_len;
        std::vector<uint8_t> decoded(K, 0);
        std::vector<uint8_t> restored(N);

        alignas(32) float probs[256];
        for (uint32_t i = 0; i < N; ++i) {
            engine.forward(&decoded[i], probs);
            uint8_t sym = decoder.decode(probs);
            decoded.push_back(sym);
            restored[i] = sym;
        }

        std::ofstream out_file(out_path, std::ios::binary);
        out_file.write(reinterpret_cast<const char*>(restored.data()), N);

        auto t1_decomp = std::chrono::high_resolution_clock::now();
        double decomp_sec = std::chrono::duration<double>(t1_decomp - t0_decomp).count();
        double kb_s = (N / 1024.0) / decomp_sec;

        std::cout << "🎉 Decompression Complete!\n"
                  << "   Restored File: " << out_path << " (" << N << " bytes)\n"
                  << "   Throughput:    " << kb_s << " KB/s (" << decomp_sec << "s)\n";

    } else if (mode == "-t") {
        std::string in_path = argv[3];
        std::ifstream in_file(in_path, std::ios::binary);
        std::vector<uint8_t> data((std::istreambuf_iterator<char>(in_file)), std::istreambuf_iterator<char>());
        size_t N = data.size();

        std::cout << "🔬 Running In-Memory Roundtrip Benchmark on " << in_path << " (" << N << " bytes)...\n";

        // Compress
        auto t0 = std::chrono::high_resolution_clock::now();
        const uint32_t K = engine.context_len;
        std::vector<uint8_t> padded(K, 0);
        padded.insert(padded.end(), data.begin(), data.end());

        std::vector<std::vector<float>> all_probs(N, std::vector<float>(256));
#pragma omp parallel for schedule(dynamic, 64)
        for (size_t i = 0; i < N; ++i) {
            engine.forward(&padded[i], all_probs[i].data());
        }

        RangeEncoder encoder;
        for (size_t i = 0; i < N; ++i) {
            encoder.encode(data[i], all_probs[i].data());
        }
        std::vector<uint8_t> payload = encoder.finish();
        auto t_comp = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - t0).count();

        // Decompress
        t0 = std::chrono::high_resolution_clock::now();
        RangeDecoder decoder;
        decoder.init(payload.data(), payload.size());

        std::vector<uint8_t> decoded(K, 0);
        std::vector<uint8_t> restored(N);

        alignas(32) float probs[256];
        for (size_t i = 0; i < N; ++i) {
            engine.forward(&decoded[i], probs);
            uint8_t sym = decoder.decode(probs);
            decoded.push_back(sym);
            restored[i] = sym;
        }
        auto t_decomp = std::chrono::duration<double>(std::chrono::high_resolution_clock::now() - t0).count();

        // Verify
        bool match = (data == restored);
        std::cout << "======================================================================\n"
                  << "📊 BENCHMARK RESULTS:\n"
                  << "   Original Size:   " << N << " bytes\n"
                  << "   Compressed Size: " << payload.size() + 4 << " bytes ("
                  << (static_cast<double>(payload.size() + 4) / N) * 100.0 << "%, "
                  << ((payload.size() + 4) * 8.0) / N << " bits/byte)\n"
                  << "   Compression:     " << (N / (1024.0 * 1024.0)) / t_comp << " MB/s (" << t_comp * 1000.0 << " ms)\n"
                  << "   Decompression:   " << (N / 1024.0) / t_decomp << " KB/s (" << t_decomp << " s)\n"
                  << "   Bit-Exact Match: " << (match ? "✅ 100% IDENTICAL" : "❌ MISMATCH") << "\n"
                  << "======================================================================\n";
    }

    return 0;
}
