#include <iostream>
#include <windows.h>
#include <iomanip>
using namespace std;

// ====================== 全局配置（仅100规模） ======================
const int TEST_SIZES[] = {100};  // 仅测试100×100规模，避免大数组
const int SIZE_COUNT = sizeof(TEST_SIZES) / sizeof(int);
const int RUN_TIMES = 100;       // 重复运行次数，保证计时精度
const int MAX_SIZE = 100;        // 最大数组大小：100×100

// ====================== 1. 矩阵列与向量内积 ======================
/**
 * 平凡算法：逐列访问（Cache命中率低）
 */
void matrix_vec_col(int n, double mat[][MAX_SIZE], double vec[], double res[]) {
    for (int col = 0; col < n; col++) {
        res[col] = 0.0;
        for (int row = 0; row < n; row++) {
            res[col] += mat[row][col] * vec[row];
        }
    }
}

/**
 * 优化算法：逐行访问（Cache命中率高）
 */
void matrix_vec_row(int n, double mat[][MAX_SIZE], double vec[], double res[]) {
    // 初始化结果数组
    for (int col = 0; col < n; col++) res[col] = 0.0;
    
    for (int row = 0; row < n; row++) {
        for (int col = 0; col < n; col++) {
            res[col] += mat[row][col] * vec[row];
        }
    }
}

// ====================== 2. n个数求和 ======================
/**
 * 平凡算法：逐个累加（链式依赖）
 */
double sum_linear(int n, double arr[]) {
    double sum = 0.0;
    for (int i = 0; i < n; i++) {
        sum += arr[i];
    }
    return sum;
}

/**
 * 优化算法：两路链式累加（无依赖）
 */
double sum_2way(int n, double arr[]) {
    double sum1 = 0.0, sum2 = 0.0;
    for (int i = 0; i < n; i += 2) {
        sum1 += arr[i];
        sum2 += arr[i+1];
    }
    return sum1 + sum2;
}

// ====================== 工具函数：高精度计时 ======================
double get_current_time() {
    LARGE_INTEGER freq, cnt;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&cnt);
    return cnt.QuadPart * 1000.0 / freq.QuadPart;
}

// ====================== 测试函数 ======================
void test_matrix_vec() {
    cout << "===== Matrix-Vector Inner Product Test (100×100) =====" << endl;
    cout << setw(8) << "Size" << setw(15) << "Col(ms)" << setw(15) << "Row(ms)" << endl;
    cout << "-----------------------------------------" << endl;

    for (int i = 0; i < SIZE_COUNT; i++) {
        int n = TEST_SIZES[i];
        // 100×100小数组，栈上分配无压力
        double mat[MAX_SIZE][MAX_SIZE], vec[MAX_SIZE], res_col[MAX_SIZE], res_row[MAX_SIZE];
        
        // 初始化数据
        for (int row = 0; row < n; row++) {
            vec[row] = row;
            for (int col = 0; col < n; col++) {
                mat[row][col] = row + col;
            }
        }

        // 测试平凡算法
        double start = get_current_time();
        for (int k = 0; k < RUN_TIMES; k++) {
            matrix_vec_col(n, mat, vec, res_col);
        }
        double time_col = (get_current_time() - start) / RUN_TIMES;

        // 测试优化算法
        start = get_current_time();
        for (int k = 0; k < RUN_TIMES; k++) {
            matrix_vec_row(n, mat, vec, res_row);
        }
        double time_row = (get_current_time() - start) / RUN_TIMES;

        // 输出结果
        cout << setw(8) << n 
             << setw(15) << fixed << setprecision(3) << time_col 
             << setw(15) << fixed << setprecision(3) << time_row << endl;
    }
    cout << endl;
}

void test_sum() {
    cout << "===== Array Sum Test (100 elements) =====" << endl;
    cout << setw(8) << "Size" << setw(15) << "Linear(ms)" << setw(15) << "2-way(ms)" << endl;
    cout << "-----------------------------------------" << endl;

    for (int i = 0; i < SIZE_COUNT; i++) {
        int n = TEST_SIZES[i];
        // 确保n为偶数
        n = n % 2 == 0 ? n : n - 1;
        double arr[MAX_SIZE];
        
        // 初始化数据
        for (int j = 0; j < n; j++) {
            arr[j] = j;
        }

        // 测试平凡算法
        double start = get_current_time();
        double sum_lin = 0.0;
        for (int k = 0; k < RUN_TIMES; k++) {
            sum_lin = sum_linear(n, arr);
        }
        double time_lin = (get_current_time() - start) / RUN_TIMES;

        // 测试优化算法
        start = get_current_time();
        double sum_2w = 0.0;
        for (int k = 0; k < RUN_TIMES; k++) {
            sum_2w = sum_2way(n, arr);
        }
        double time_2w = (get_current_time() - start) / RUN_TIMES;

        // 输出结果
        cout << setw(8) << n 
             << setw(15) << fixed << setprecision(3) << time_lin 
             << setw(15) << fixed << setprecision(3) << time_2w << endl;
    }
    cout << endl;
}

// ====================== 主函数 ======================
int main() {
    // 直接执行测试，无编码/暂停干扰
    test_matrix_vec();
    test_sum();
    return 0;
}