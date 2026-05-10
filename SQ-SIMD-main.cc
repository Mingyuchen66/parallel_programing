#include <vector>
#include <cstring>
#include <string>
#include <iostream>
#include <fstream>
#include <set>
#include <sys/time.h>
#include <omp.h>
#include <algorithm>
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
        appr_alg->addPoint(base + i*vecdim, i);
    }

    char path_index[1024] = "files/hnsw.index";
    appr_alg->saveIndex(path_index);
}

// 原始自动向量化版本
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

// ======================================================================
// ======================== SQ-SIMD 纯实现 ==============================
// ======================================================================

void quantize_to_int8(const float* src, int8_t* dst, int dim, float scale) {
    for (int i = 0; i < dim; i++) {
        float v = src[i] * scale;
        if (v > 127.0f) v = 127.0f;
        if (v < -128.0f) v = -128.0f;
        dst[i] = (int8_t)v;
    }
}

float dot_sq_simd(const float* a, const float* b, int dim) {
    const float scale = 127.0f;
    int8_t qa[96], qb[96];

    quantize_to_int8(a, qa, dim, scale);
    quantize_to_int8(b, qb, dim, scale);

    int sum = 0;
    for (int i = 0; i < dim; i++) {
        sum += qa[i] * qb[i];
    }

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

// ======================================================================

int main(int argc, char *argv[])
{
    size_t test_number = 0, base_number = 0;
    size_t test_gt_d = 0, vecdim = 0;

    std::string data_path = "/anndata/";
    auto test_query = LoadData<float>(data_path + "DEEP100K.query.fbin", test_number, vecdim);
    auto test_gt = LoadData<int>(data_path + "DEEP100K.gt.query.100k.top100.bin", test_number, test_gt_d);
    auto base = LoadData<float>(data_path + "DEEP100K.base.100k.fbin", base_number, vecdim);
    test_number = 2000;

    const size_t k = 10;
    std::vector<SearchResult> results(test_number);

    for(int i = 0; i < test_number; ++i) {
        const unsigned long Converter = 1000 * 1000;
        struct timeval val;
        gettimeofday(&val, NULL);

        // ===================== 运行 SQ-SIMD =====================
        auto res = flat_search_sq_simd(base, test_query + i*vecdim, base_number, vecdim, k);
        // ========================================================

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

    std::cout << "=== SQ-SIMD 版本结果 ===" << std::endl;
    std::cout << "average recall: " << avg_recall / test_number << std::endl;
    std::cout << "average latency (us): " << avg_latency / test_number << std::endl;

    delete[] test_query;
    delete[] test_gt;
    delete[] base;

    return 0;
}