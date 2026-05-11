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
#include <queue>
#include <cmath>
#include <algorithm>
#include <arm_neon.h> // 引入ARM NEON指令集
#include "hnswlib/hnswlib/hnswlib.h"

// 如果你的本地有 flat_scan.h 可以保留，不需要的话可以注释掉
// #include "flat_scan.h" 

using namespace hnswlib;

template<typename T>
T *LoadData(std::string data_path, size_t& n, size_t& d)
{
    std::ifstream fin;
    fin.open(data_path, std::ios::in | std::ios::binary);
    if (!fin.is_open()) {
        std::cerr << "Failed to open " << data_path << "\n";
        exit(1);
    }
    fin.read((char*)&n, 4);
    fin.read((char*)&d, 4);
    T* data = new T[n * d];
    int sz = sizeof(T);
    for(size_t i = 0; i < n; ++i){
        fin.read(((char*)data + i * d * sz), d * sz);
    }
    fin.close();

    std::cerr << "load data " << data_path << "\n";
    std::cerr << "dimension: " << d << "  number:" << n << "  size_per_element:" << sizeof(T) << "\n";

    return data;
}

struct SearchResult
{
    float recall;
    int64_t latency; // 单位us
};

// ==========================================================
// SIMD L2距离计算 (用于Rerank精确计算)
// ==========================================================
inline float L2Sqr_SIMD(const float* a, const float* b, size_t d) {
    float32x4_t sum = vdupq_n_f32(0.0f);
    size_t i = 0;
    for (; i + 3 < d; i += 4) {
        float32x4_t va = vld1q_f32(a + i);
        float32x4_t vb = vld1q_f32(b + i);
        float32x4_t diff = vsubq_f32(va, vb);
        sum = vmlaq_f32(sum, diff, diff);
    }
    float res[4];
    vst1q_f32(res, sum);
    float final_dist = res[0] + res[1] + res[2] + res[3];
    // 处理剩余维度
    for (; i < d; ++i) {
        float diff = a[i] - b[i];
        final_dist += diff * diff;
    }
    return final_dist;
}

// ==========================================================
// PQ-SIMD 索引类 (结合Rerank解决Recall低的问题)
// ==========================================================
class PQIndex {
public:
    size_t N, D;
    size_t M;      // 子空间数量 (e.g., 24)
    size_t Ks;     // 每个子空间的聚类中心数 (e.g., 256)
    size_t Ds;     // 子空间维度 (D / M = 4)

    std::vector<float> centroids; // 大小: M * Ks * Ds
    std::vector<uint8_t> codes;   // 大小: N * M

    PQIndex(size_t base_number, size_t vecdim, size_t m = 24, size_t ks = 256) 
        : N(base_number), D(vecdim), M(m), Ks(ks), Ds(vecdim / m) {
        centroids.resize(M * Ks * Ds);
        codes.resize(N * M);
    }

    // 训练KMeans获取Codebook
    void train_and_encode(const float* base) {
        std::cerr << "Start training PQ (M=" << M << ", Ks=" << Ks << ")...\n";
        const int max_iters = 15;

        // 并行训练每个子空间
        #pragma omp parallel for
        for (size_t m = 0; m < M; ++m) {
            float* cur_centroids = &centroids[m * Ks * Ds];
            
            // 随机初始化聚类中心
            for (size_t k = 0; k < Ks; ++k) {
                size_t rand_idx = (rand() % N);
                for (size_t d = 0; d < Ds; ++d) {
                    cur_centroids[k * Ds + d] = base[rand_idx * D + m * Ds + d];
                }
            }

            std::vector<int> assign(N, 0);
            std::vector<float> new_cents(Ks * Ds, 0.0f);
            std::vector<int> counts(Ks, 0);

            for (int iter = 0; iter < max_iters; ++iter) {
                std::fill(new_cents.begin(), new_cents.end(), 0.0f);
                std::fill(counts.begin(), counts.end(), 0);

                // E-step: 分配最近中心点
                for (size_t i = 0; i < N; ++i) {
                    const float* sub_vec = &base[i * D + m * Ds];
                    float min_dist = 1e30f;
                    int best_k = 0;
                    for (size_t k = 0; k < Ks; ++k) {
                        float dist = 0;
                        for(size_t d = 0; d < Ds; ++d) {
                            float diff = sub_vec[d] - cur_centroids[k * Ds + d];
                            dist += diff * diff;
                        }
                        if (dist < min_dist) { min_dist = dist; best_k = k; }
                    }
                    assign[i] = best_k;
                    counts[best_k]++;
                    for (size_t d = 0; d < Ds; ++d) {
                        new_cents[best_k * Ds + d] += sub_vec[d];
                    }
                }

                // M-step: 更新中心点
                for (size_t k = 0; k < Ks; ++k) {
                    if (counts[k] > 0) {
                        for (size_t d = 0; d < Ds; ++d) {
                            cur_centroids[k * Ds + d] = new_cents[k * Ds + d] / counts[k];
                        }
                    }
                }
            }
            
            // 最终编码
            for (size_t i = 0; i < N; ++i) {
                codes[i * M + m] = assign[i];
            }
        }
        std::cerr << "PQ training and encoding finished.\n";
    }

    bool load_index(const std::string& prefix) {
        std::ifstream fc(prefix + "_centroids.bin", std::ios::binary);
        std::ifstream fd(prefix + "_codes.bin", std::ios::binary);
        if (fc.is_open() && fd.is_open()) {
            fc.read((char*)centroids.data(), centroids.size() * sizeof(float));
            fd.read((char*)codes.data(), codes.size() * sizeof(uint8_t));
            return true;
        }
        return false;
    }

    void save_index(const std::string& prefix) {
        std::ofstream fc(prefix + "_centroids.bin", std::ios::binary);
        std::ofstream fd(prefix + "_codes.bin", std::ios::binary);
        fc.write((char*)centroids.data(), centroids.size() * sizeof(float));
        fd.write((char*)codes.data(), codes.size() * sizeof(uint8_t));
    }

    // PQ 结合 SIMD Rerank 搜索
    std::priority_queue<std::pair<float, int>> search(const float* query, const float* base, size_t k) {
        // 1. 利用SIMD构造查询专属的LUT (Look-Up Table)
        std::vector<float> LUT(M * Ks);
        for (size_t m = 0; m < M; ++m) {
            const float* q_sub = query + m * Ds;
            for (size_t c = 0; c < Ks; ++c) {
                const float* cent = &centroids[m * Ks * Ds + c * Ds];
                
                // M=24 时 Ds=4, 恰好适合一次NEON读取
                float32x4_t vq = vld1q_f32(q_sub);
                float32x4_t vc = vld1q_f32(cent);
                float32x4_t diff = vsubq_f32(vq, vc);
                float32x4_t sq = vmulq_f32(diff, diff);
                
                float res[4];
                vst1q_f32(res, sq);
                LUT[m * Ks + c] = res[0] + res[1] + res[2] + res[3];
            }
        }

        // 2. 使用PQ扫描所有数据库向量 (维护一个大小为 rerank_k 的最大堆)
        size_t rerank_k = 400; // 为解决Recall过低，提取前400名进行重排
        std::priority_queue<std::pair<float, int>> pq_cands;

        // 采用4路循环展开加速查表累加过程
        for (size_t i = 0; i < N; i += 4) {
            float d0 = 0, d1 = 0, d2 = 0, d3 = 0;
            const uint8_t* c0 = &codes[i * M];
            const uint8_t* c1 = &codes[(i + 1) * M];
            const uint8_t* c2 = &codes[(i + 2) * M];
            const uint8_t* c3 = &codes[(i + 3) * M];

            for (size_t m = 0; m < M; ++m) {
                const float* lut_m = &LUT[m * Ks];
                d0 += lut_m[c0[m]];
                d1 += lut_m[c1[m]];
                d2 += lut_m[c2[m]];
                d3 += lut_m[c3[m]];
            }

            // 处理结果 d0
            if (pq_cands.size() < rerank_k) { pq_cands.push({d0, i}); }
            else if (d0 < pq_cands.top().first) { pq_cands.pop(); pq_cands.push({d0, i}); }
            // 处理结果 d1
            if (i+1 < N) {
                if (pq_cands.size() < rerank_k) { pq_cands.push({d1, i+1}); }
                else if (d1 < pq_cands.top().first) { pq_cands.pop(); pq_cands.push({d1, i+1}); }
            }
            // 处理结果 d2
            if (i+2 < N) {
                if (pq_cands.size() < rerank_k) { pq_cands.push({d2, i+2}); }
                else if (d2 < pq_cands.top().first) { pq_cands.pop(); pq_cands.push({d2, i+2}); }
            }
            // 处理结果 d3
            if (i+3 < N) {
                if (pq_cands.size() < rerank_k) { pq_cands.push({d3, i+3}); }
                else if (d3 < pq_cands.top().first) { pq_cands.pop(); pq_cands.push({d3, i+3}); }
            }
        }

        // 3. 精确距离重排 (SIMD Rerank)
        std::priority_queue<std::pair<float, int>> final_topK;
        while (!pq_cands.empty()) {
            int cand_idx = pq_cands.top().second;
            pq_cands.pop();

            // 使用 SIMD 精确计算候选集与 Query 的距离
            float exact_dist = L2Sqr_SIMD(query, base + cand_idx * D, D);
            
            if (final_topK.size() < k) {
                final_topK.push({exact_dist, cand_idx});
            } else if (exact_dist < final_topK.top().first) {
                final_topK.pop();
                final_topK.push({exact_dist, cand_idx});
            }
        }

        return final_topK;
    }
};


int main(int argc, char *argv[])
{
    size_t test_number = 0, base_number = 0;
    size_t test_gt_d = 0, vecdim = 0;

    std::string data_path = "/anndata/";
    auto test_query = LoadData<float>(data_path + "DEEP100K.query.fbin", test_number, vecdim);
    auto test_gt = LoadData<int>(data_path + "DEEP100K.gt.query.100k.top100.bin", test_number, test_gt_d);
    auto base = LoadData<float>(data_path + "DEEP100K.base.100k.fbin", base_number, vecdim);
    
    // 只测试前2000条查询
    test_number = 2000;
    const size_t k = 10;

    std::vector<SearchResult> results;
    results.resize(test_number);

    // ==========================================
    // 构建或加载 PQ-SIMD 索引
    // ==========================================
    PQIndex pq_index(base_number, vecdim, 24, 256); // M=24, Ks=256
    std::string pq_path = "files/pq";
    if (!pq_index.load_index(pq_path)) {
        pq_index.train_and_encode(base);
        // 如果文件目录没有 files/ 可能保存失败，但不影响当次运行
        pq_index.save_index(pq_path); 
    } else {
        std::cerr << "Successfully loaded PQ index from " << pq_path << "\n";
    }

    // ==========================================
    // 查询测试代码
    // ==========================================
    for(size_t i = 0; i < test_number; ++i) {
        const unsigned long Converter = 1000 * 1000;
        struct timeval val;
        int ret = gettimeofday(&val, NULL);

        // 调用 PQ-SIMD-Rerank 检索算法
        auto res = pq_index.search(test_query + i * vecdim, base, k);

        struct timeval newVal;
        ret = gettimeofday(&newVal, NULL);
        int64_t diff = (newVal.tv_sec * Converter + newVal.tv_usec) - (val.tv_sec * Converter + val.tv_usec);

        std::set<uint32_t> gtset;
        for(size_t j = 0; j < k; ++j){
            int t = test_gt[j + i * test_gt_d];
            gtset.insert(t);
        }

        size_t acc = 0;
        while (res.size()) {
            int x = res.top().second;
            if(gtset.find(x) != gtset.end()){
                ++acc;
            }
            res.pop();
        }
        float recall = (float)acc / k;

        results[i] = {recall, diff};
    }

    float avg_recall = 0, avg_latency = 0;
    for(size_t i = 0; i < test_number; ++i) {
        avg_recall += results[i].recall;
        avg_latency += results[i].latency;
    }

    std::cout << "average recall: " << avg_recall / test_number << "\n";
    std::cout << "average latency (us): " << avg_latency / test_number << "\n";
    
    delete[] test_query;
    delete[] test_gt;
    delete[] base;
    return 0;
}
