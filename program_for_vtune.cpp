#include <iostream>
#include <windows.h>
#include <iomanip>
#include <cstring>
using namespace std;

const int MAX_SIZE = 1000;
const int TEST_SIZES[] = {100, 500, 1000};
const int SIZE_COUNT = sizeof(TEST_SIZES)/sizeof(int);

// 求和必须超级多次，才能让 VTune 采样正常
const int RUN_TIMES = 300000;

// 矩阵-向量内积：平凡算法（逐列，Cache 不友好）
void matrix_vec_col(int n, double** mat, double vec[], double res[]) {
    for (int col = 0; col < n; col++) {
        res[col] = 0.0;
        for (int row = 0; row < n; row++) {
            res[col] += mat[row][col] * vec[row];
        }
    }
}

// 矩阵-向量内积：优化算法（逐行，Cache 友好）
void matrix_vec_row(int n, double** mat, double vec[], double res[]) {
    for (int col = 0; col < n; col++)
        res[col] = 0.0;

    for (int row = 0; row < n; row++) {
        for (int col = 0; col < n; col++) {
            res[col] += mat[row][col] * vec[row];
        }
    }
}

// 求和：平凡算法（有数据依赖）
double sum_linear(int n, double arr[]) {
    double sum = 0.0;
    for (int i = 0; i < n; i++)
        sum += arr[i];

    // 防止编译器优化掉循环
    if (sum < 0) cout << sum;
    return sum;
}

// 求和：优化算法（两路并行，消除依赖）
double sum_2way(int n, double arr[]) {
    double sum1 = 0.0, sum2 = 0.0;
    for (int i = 0; i < n - 1; i += 2) {
        sum1 += arr[i];
        sum2 += arr[i + 1];
    }
    if (n % 2 == 1)
        sum1 += arr[n - 1];

    double total = sum1 + sum2;
    // 防止编译器优化掉循环
    if (total < 0) cout << total;
    return total;
}

// 高精度计时
double get_current_time() {
    LARGE_INTEGER freq, cnt;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&cnt);
    return cnt.QuadPart * 1000.0 / freq.QuadPart;
}

// 测试矩阵 col（平凡）
void run_matrix_col() {
    cout << "=== 矩阵内积 平凡算法 col ===" << endl;
    cout << setw(8) << "规模" << setw(15) << "平均耗时(ms)" << endl;
    cout << "-----------------------------------------" << endl;

    for (int s = 0; s < SIZE_COUNT; s++) {
        int n = TEST_SIZES[s];

        double** mat = new double*[n];
        for (int i = 0; i < n; i++)
            mat[i] = new double[n];

        double* vec = new double[n];
        double* res = new double[n];

        for (int row = 0; row < n; row++) {
            vec[row] = row;
            for (int col = 0; col < n; col++)
                mat[row][col] = row + col;
        }

        double start = get_current_time();
        for (int k = 0; k < RUN_TIMES; k++)
            matrix_vec_col(n, mat, vec, res);
        double avg = (get_current_time() - start) / RUN_TIMES;

        cout << setw(8) << n << setw(15) << fixed << setprecision(4) << avg << endl;

        for (int i = 0; i < n; i++) delete[] mat[i];
        delete[] mat;
        delete[] vec;
        delete[] res;
    }
    cout << endl;
}

// 测试矩阵 row（优化）
void run_matrix_row() {
    cout << "=== 矩阵内积 优化算法 row ===" << endl;
    cout << setw(8) << "规模" << setw(15) << "平均耗时(ms)" << endl;
    cout << "-----------------------------------------" << endl;

    for (int s = 0; s < SIZE_COUNT; s++) {
        int n = TEST_SIZES[s];

        double** mat = new double*[n];
        for (int i = 0; i < n; i++)
            mat[i] = new double[n];

        double* vec = new double[n];
        double* res = new double[n];

        for (int row = 0; row < n; row++) {
            vec[row] = row;
            for (int col = 0; col < n; col++)
                mat[row][col] = row + col;
        }

        double start = get_current_time();
        for (int k = 0; k < RUN_TIMES; k++)
            matrix_vec_row(n, mat, vec, res);
        double avg = (get_current_time() - start) / RUN_TIMES;

        cout << setw(8) << n << setw(15) << fixed << setprecision(4) << avg << endl;

        for (int i = 0; i < n; i++) delete[] mat[i];
        delete[] mat;
        delete[] vec;
        delete[] res;
    }
    cout << endl;
}

// 测试求和 linear
void run_sum_linear() {
    cout << "=== 求和 平凡算法 linear ===" << endl;
    cout << setw(8) << "规模" << setw(15) << "平均耗时(ms)" << endl;
    cout << "-----------------------------------------" << endl;

    for (int s = 0; s < SIZE_COUNT; s++) {
        int n = TEST_SIZES[s];
        double* arr = new double[n];
        for (int i = 0; i < n; i++) arr[i] = i;

        double start = get_current_time();
        for (int k = 0; k < RUN_TIMES; k++)
            sum_linear(n, arr);
        double avg = (get_current_time() - start) / RUN_TIMES;

        cout << setw(8) << n << setw(15) << fixed << setprecision(4) << avg << endl;
        delete[] arr;
    }
    cout << endl;
}

// 测试求和 2way
void run_sum_2way() {
    cout << "=== 求和 优化算法 2way ===" << endl;
    cout << setw(8) << "规模" << setw(15) << "平均耗时(ms)" << endl;
    cout << "-----------------------------------------" << endl;

    for (int s = 0; s < SIZE_COUNT; s++) {
        int n = TEST_SIZES[s];
        double* arr = new double[n];
        for (int i = 0; i < n; i++) arr[i] = i;

        double start = get_current_time();
        for (int k = 0; k < RUN_TIMES; k++)
            sum_2way(n, arr);
        double avg = (get_current_time() - start) / RUN_TIMES;

        cout << setw(8) << n << setw(15) << fixed << setprecision(4) << avg << endl;
        delete[] arr;
    }
    cout << endl;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        cout << "Usage: lab1.exe col|row|linear|2way" << endl;
        return 0;
    }

    if (strcmp(argv[1], "col") == 0) run_matrix_col();
    if (strcmp(argv[1], "row") == 0) run_matrix_row();
    if (strcmp(argv[1], "linear") == 0) run_sum_linear();
    if (strcmp(argv[1], "2way") == 0) run_sum_2way();

    return 0;
}