#include <iostream>
#include <windows.h>
#include <iomanip>
#include <cstring>
using namespace std;

const int MAX_SIZE = 1000;
const int TEST_SIZES[] = {100, 500, 1000};
const int SIZE_COUNT = sizeof(TEST_SIZES)/sizeof(int);
const int RUN_TIMES = 100000;

void matrix_vec_col(int n, double** mat, double vec[], double res[]) {
    for (int col = 0; col < n; col++) {
        res[col] = 0.0;
        for (int row = 0; row < n; row++) {
            res[col] += mat[row][col] * vec[row];
        }
    }
}

void matrix_vec_row(int n, double** mat, double vec[], double res[]) {
    for (int col = 0; col < n; col++)
        res[col] = 0.0;
    for (int row = 0; row < n; row++) {
        for (int col = 0; col < n; col++) {
            res[col] += mat[row][col] * vec[row];
        }
    }
}

double sum_linear(int n, double arr[]) {
    double sum = 0.0;
    for (int i = 0; i < n; i++)
        sum += arr[i];
    if (sum < 0) cout << sum;
    return sum;
}

double sum_2way(int n, double arr[]) {
    double sum1 = 0.0, sum2 = 0.0;
    for (int i = 0; i < n - 1; i += 2) {
        sum1 += arr[i];
        sum2 += arr[i + 1];
    }
    if (n % 2 == 1)
        sum1 += arr[n - 1];
    double total = sum1 + sum2;
    if (total < 0) cout << total;
    return total;
}

double get_current_time() {
    LARGE_INTEGER freq, cnt;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&cnt);
    return cnt.QuadPart * 1000.0 / freq.QuadPart;
}

void run_matrix_col() {
    cout << "=== 矩阵内积 平凡算法(col) ===" << endl;
    cout << setw(8) << "规模" << setw(15) << "平均耗时(ms)" << endl;
    cout << "-----------------------------------------" << endl;
    for (int s = 0; s < SIZE_COUNT; s++) {
        int n = TEST_SIZES[s];
        double** mat = new double*[n];
        for (int i = 0; i < n; i++) mat[i] = new double[n];
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
        delete[] mat; delete[] vec; delete[] res;
    }
    cout << endl;
}

void run_matrix_row() {
    cout << "=== 矩阵内积 优化算法(row) ===" << endl;
    cout << setw(8) << "规模" << setw(15) << "平均耗时(ms)" << endl;
    cout << "-----------------------------------------" << endl;
    for (int s = 0; s < SIZE_COUNT; s++) {
        int n = TEST_SIZES[s];
        double** mat = new double*[n];
        for (int i = 0; i < n; i++) mat[i] = new double[n];
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
        delete[] mat; delete[] vec; delete[] res;
    }
    cout << endl;
}

void run_sum_linear() {
    cout << "=== 求和 平凡算法(linear) ===" << endl;
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

void run_sum_2way() {
    cout << "=== 求和 优化算法(2way) ===" << endl;
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
