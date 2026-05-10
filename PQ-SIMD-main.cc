#include <vector>
#include <cstring>
#include <string>
#include <iostream>
#include <fstream>
#include <set>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <sys/time.h>
#include <omp.h>
#include <algorithm>
#include <arm_neon.h>
#include "hnswlib/hnswlib/hnswlib.h"
#include "flat_scan.h"

using namespace hnswlib;

template<typename T>
T *LoadData(std::string data_path, size_t& n, size_t& d)
{
    std::ifstream fin;
    fin.open(data_path, std::ios::in | std::ios::binary);
    fin.read((char*)&n,4);
    fin.read((char*)&d,4);
    T* data = new T[n*d];
    int sz = sizeof(T);
    for(int i = 0; i < n; ++i){
        fin.read(((char*)data + i*d*sz), d*sz);
    }
    fin.close();

    std::cerr<<"load data "<<data_path<<"\n";
    std::cerr<<"dimension: "<<d<<"  number:"<<n<<"  size_per_element:"<<sizeof(T)<<"\n";

    return data;
}

struct SearchResult
{
    float recall;
    int64_t latency;
};

void build_index(float* base, size_t base_number, size_t vecdim)
{
    const int efConstruction = 150;
    const int M = 16;

    HierarchicalNSW<float> *appr_alg;
    InnerProductSpace ipspace(vecdim);
    appr_alg = new HierarchicalNSW<float>(&ipspace, base_number, M, efConstruction);

    appr_alg->addPoint(base, 0);
    #pragma omp parallel for
    for(int i = 1; i < base_number; ++i) {
        appr_alg->addPoint(base + 1ll*vecdim*i, i);
    }

    char path_index[1024] = "files/hnsw.index";
    appr_alg->saveIndex(path_index);
}

// 编译器自动向量化版本
std::priority_queue<std::pair<float, uint32_t>>
flat_search_opt(float* base, float* query, size_t base_number, size_t vecdim, size_t k) {
    std::priority_queue<std::pair<float, uint32_t>> q;
    for(int i = 0; i < base_number; ++i) {
        float dis = 0.0f;
        const float* b = base + i * vecdim;
        const float* qry = query;

        for(int d = 0; d < vecdim; ++d) {
            dis += b[d] * qry[d];
        }
        dis = 1.0f - dis;

        if(q.size() < k) {
            q.push(std::make_pair(dis, i));
        } else {
            if(dis < q.top().first) {
                q.pop();
                q.push(std::make_pair(dis, i));
            }
        }
    }
    return q;
}

// ======================== SQ-SIMD（ARM NEON） ========================
void quantize_to_int8(const float* src, int8_t* dst, int dim, float scale) {
    for (int i = 0; i < dim; i++) {
        float v = src[i] * scale;
        v = std::max(std::min(v, 127.0f), -128.0f);
        dst[i] = (int8_t)v;
    }
}

float dot_sq_simd(const float* a, const float* b, int dim) {
    const float scale = 127.0f;
    int8_t qa[96], qb[96];
    quantize_to_int8(a, qa, dim, scale);
    quantize_to_int8(b, qb, dim, scale);

    int32x4_t sum_vec = vdupq_n_s32(0);
    int i = 0;

    for (; i + 16 <= dim; i += 16) {
        int8x16_t va = vld1q_s8(qa + i);
        int8x16_t vb = vld1q_s8(qb + i);
        int16x8_t va16 = vmovl_s8(vget_low_s8(va));
        int16x8_t vb16 = vmovl_s8(vget_low_s8(vb));
        int32x4_t va32_lo = vmovl_s16(vget_low_s16(va16));
        int32x4_t vb32_lo = vmovl_s16(vget_low_s16(vb16));
        sum_vec = vmlaq_s32(sum_vec, va32_lo, vb32_lo);

        int16x8_t va16_hi = vmovl_s8(vget_high_s8(va));
        int16x8_t vb16_hi = vmovl_s8(vget_high_s8(vb));
        int32x4_t va32_hi = vmovl_s16(vget_high_s16(va16_hi));
        int32x4_t vb32_hi = vmovl_s16(vget_high_s16(vb16_hi));
        sum_vec = vmlaq_s32(sum_vec, va32_hi, vb32_hi);
    }

    int32_t sum = vaddvq_s32(sum_vec);
    for (; i < dim; i++) sum += qa[i] * qb[i];

    return 1.0f - (float)sum / (scale * scale);
}

std::priority_queue<std::pair<float, uint32_t>>
flat_search_sq_simd(float* base, float* query, size_t base_number, size_t vecdim, size_t k) {
    std::priority_queue<std::pair<float, uint32_t>> q;

    for (size_t i = 0; i < base_number; ++i) {
        float d = dot_sq_simd(query, base + i * vecdim, vecdim);
        if (q.size() < k) {
            q.push({d, (uint32_t)i});
        } else {
            if (d < q.top().first) {
                q.pop();
                q.push({d, (uint32_t)i});
            }
        }
    }
    return q;
}

// ===================== PQ-SIMD（ARM NEON） ===========================
#define PQ_M        4
#define PQ_D_SUB    24
#define PQ_CENTERS  256

float pq_codebook[PQ_M][PQ_CENTERS][PQ_D_SUB] = {0};

void init_pq_codebook() {
    for (int m = 0; m < PQ_M; m++) {
        for (int c = 0; c < PQ_CENTERS; c++) {
            for (int d = 0; d < PQ_D_SUB; d++) {
                pq_codebook[m][c][d] = (c % 127) / 127.0f;
            }
        }
    }
}

void pq_encode_vector(const float* vec, uint8_t* code) {
    for (int m = 0; m < PQ_M; m++) {
        const float* sub = vec + m * PQ_D_SUB;
        int best = 0;
        float min_dist = 1e9;
        for (int c = 0; c < PQ_CENTERS; c++) {
            float dist = 0.0f;
            for (int d = 0; d < PQ_D_SUB; d++) {
                dist += sub[d] * pq_codebook[m][c][d];
            }
            dist = 1.0f - dist;
            if (dist < min_dist) {
                min_dist = dist;
                best = c;
            }
        }
        code[m] = best;
    }
}

uint8_t* pq_codes = nullptr;

void pq_build_codes(float* base, size_t base_num, int dim) {
    pq_codes = new uint8_t[base_num * PQ_M];
    for (size_t i = 0; i < base_num; i++) {
        pq_encode_vector(base + i * dim, pq_codes + i * PQ_M);
    }
}

void pq_build_lut(const float* query, float lut[PQ_M][PQ_CENTERS]) {
    for (int m = 0; m < PQ_M; m++) {
        const float* q_sub = query + m * PQ_D_SUB;
        for (int c = 0; c < PQ_CENTERS; c++) {
            float ip = 0.0f;
            for (int d = 0; d < PQ_D_SUB; d++) {
                ip += q_sub[d] * pq_codebook[m][c][d];
            }
            lut[m][c] = 1.0f - ip;
        }
    }
}

float pq_dist_simd(const uint8_t* code, const float lut[PQ_M][PQ_CENTERS]) {
    float32x4_t sum_vec = vdupq_n_f32(0.0f);
    for (int m = 0; m < PQ_M; m++) {
        int c = code[m];
        float val = lut[m][c];
        sum_vec = vaddq_f32(sum_vec, vdupq_n_f32(val));
    }
    float buf[4];
    vst1q_f32(buf, sum_vec);
    return buf[0] + buf[1] + buf[2] + buf[3];
}

std::priority_queue<std::pair<float, uint32_t>>
flat_search_pq_simd(float* base, float* query, size_t base_number, size_t vecdim, size_t k) {
    std::priority_queue<std::pair<float, uint32_t>> q;
    static float lut[PQ_M][PQ_CENTERS];

    pq_build_lut(query, lut);

    for (size_t i = 0; i < base_number; i++) {
        float d = pq_dist_simd(pq_codes + i * PQ_M, lut);
        if (q.size() < k) {
            q.push({d, (uint32_t)i});
        } else {
            if (d < q.top().first) {
                q.pop();
                q.push({d, (uint32_t)i});
            }
        }
    }
    return q;
}

// ========================================================

int main(int argc, char *argv[])
{
    size_t test_number = 0, base_number = 0;
    size_t test_gt_d = 0, vecdim = 0;

    std::string data_path = "/anndata/";
    auto test_query = LoadData<float>(data_path + "DEEP100K.query.fbin", test_number, vecdim);
    auto test_gt = LoadData<int>(data_path + "DEEP100K.gt.query.100k.top100.bin", test_number, test_gt_d);
    auto base = LoadData<float>(data_path + "DEEP100K.base.100k.fbin", base_number, vecdim);
    test_number = 2000;

    init_pq_codebook();
    pq_build_codes(base, base_number, vecdim);

    const size_t k = 10;
    std::vector<SearchResult> results;
    results.resize(test_number);

    for(int i = 0; i < test_number; ++i) {
        const unsigned long Converter = 1000 * 1000;
        struct timeval val;
        gettimeofday(&val, NULL);

        // ==================== 运行 PQ-SIMD ====================
        auto res = flat_search_pq_simd(base, test_query + i*vecdim, base_number, vecdim, k);
        // =====================================================

        struct timeval newVal;
        gettimeofday(&newVal, NULL);
        int64_t diff = (newVal.tv_sec * Converter + newVal.tv_usec) - (val.tv_sec * Converter + val.tv_usec);

        std::set<uint32_t> gtset;
        for(int j = 0; j < k; ++j){
            int t = test_gt[j + i*test_gt_d];
            gtset.insert(t);
        }

        size_t acc = 0;
        while (res.size()) {
            int x = res.top().second;
            if(gtset.count(x)) acc++;
            res.pop();
        }
        float recall = (float)acc / k;
        results[i] = {recall, diff};
    }

    float avg_recall = 0, avg_latency = 0;
    for(int i = 0; i < test_number; ++i) {
        avg_recall += results[i].recall;
        avg_latency += results[i].latency;
    }

    std::cout << "=== PQ-SIMD 版本结果 ===\n";
    std::cout << "average recall: " << avg_recall / test_number << "\n";
    std::cout << "average latency (us): " << avg_latency / test_number << "\n";

    delete[] test_query;
    delete[] test_gt;
    delete[] base;
    delete[] pq_codes;

    return 0;
}