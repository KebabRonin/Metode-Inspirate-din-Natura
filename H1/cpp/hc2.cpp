#include <cmath>
#include <iostream>
#include <random>
#include <vector>
#include <chrono>
#include <algorithm>
#include "functions.cpp"
// #include "tqdm/tqdm.h"

#define Bitstring std::vector<bool>
#define ParameterList std::vector<double>

struct ProblemSpec {
    const OptimizationFunction& f;
    const int precision;
    const int input_dims;

    // Inferred
    int sol_len;
    int parameter_len;
    const double b_min;
    const double b_max;

    ProblemSpec(
        const OptimizationFunction& func_obj,
        const int precision,
        const int input_dims
    ) : f(func_obj),
        precision(precision),
        input_dims(input_dims),
        b_min(func_obj.bounds().first),
        b_max(func_obj.bounds().second)
    {
        double N = ((b_max - b_min) * std::pow(10, precision));
        parameter_len = static_cast<int>(std::ceil(std::log2(N)));
        sol_len = input_dims * parameter_len;
    }

    Bitstring randomBitstring() const {
        Bitstring bitstring(sol_len);
        static std::random_device rd;
        static std::mt19937 gen(rd());
        static std::uniform_int_distribution<> dis(0, 1);
        std::generate(bitstring.begin(), bitstring.end(), [&]() { return dis(gen); });
        return bitstring;
    }

    double numeric(const Bitstring& segment) const {
        int bit_value = 0;
        int power = 1;
        for (int i = 0; i < parameter_len; ++i) {
            bit_value += power * segment[i];
            power <<= 1;
        }
        return b_min + (bit_value * (b_max - b_min)) / (power / 2 - 1);
    }

    ParameterList decodeSolution(const Bitstring& bitstring) const {
        ParameterList decoded_values;
        for (int i = 0; i < input_dims; ++i) {
            auto begin = bitstring.begin() + i * parameter_len;
            auto end = begin + parameter_len;
            Bitstring segment(begin, end);
            decoded_values.push_back(numeric(segment));
        }
        return decoded_values;
    }
};

double OptimizationFunction::apply(const Bitstring& vc, const ProblemSpec& problem) const {
    return apply(problem.decodeSolution(vc));
}
double OptimizationFunction::compute(const Bitstring& vc, const ProblemSpec& problem) const {
    return compute(problem.decodeSolution(vc));
}

// Improves the given solution by evaluating neighbors and selecting the best one
Bitstring improve(Bitstring& vc, const ProblemSpec& problem) {
    double best_score = problem.f.compute(vc, problem);
    int best_neighbor = -1;

    for (size_t i = 0; i < vc.size(); ++i) {
        vc[i] = !vc[i];
        double neighbor_score = problem.f.compute(vc, problem);
        if (neighbor_score < best_score) {
            best_score = neighbor_score;
            best_neighbor = i;
        }
        vc[i] = !vc[i];
    }

    if (best_neighbor != -1) {
        vc[best_neighbor] = !vc[best_neighbor];
    }
    return vc;
}

std::pair<ParameterList, double> hillClimbing(const OptimizationFunction& f, const ProblemSpec& problem, const int max_iter) {
    Bitstring best = problem.randomBitstring();
    double best_score = f.compute(best, problem);

    for (int t = 0; t < max_iter; ++t) {
        Bitstring vc = problem.randomBitstring();
        double vc_score = f.compute(vc, problem);
        bool local = false;

        while (!local) {
            Bitstring vn = improve(vc, problem);
            double vn_score = f.compute(vn, problem);

            if (vn_score < vc_score) {
                vc = vn;
                vc_score = vn_score;
            } else {
                local = true;
            }
        }

        if (vc_score < best_score) {
            best = vc;
            best_score = vc_score;
        }
    }

    ParameterList best_decoded = problem.decodeSolution(best);
    return {best_decoded, best_score};
}

int main() {
    int n_dim = 30;
    int max_iter = 100;
    int precision = 5;
    std::cout << "Running Hill Climbing: " << "n_dim=" << n_dim << ", max_iter=" << max_iter << ", precision=" << precision << "\n";

    for (const OptimizationFunction* func_obj : FUNCTIONS) {
        // Count time
        auto start = std::chrono::high_resolution_clock::now();
        ProblemSpec problem(*func_obj, precision, n_dim);
        auto [best_solution, best_score] = hillClimbing(*func_obj, problem, max_iter);
        // Count time
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        std::printf("====\n(%s)\tElapsed time: %fs\nBest Score: %f\nBest Solution: [", problem.f.name().c_str(), elapsed.count(), best_score);
        for (double val : best_solution) {
            std::cout << val << " ";
        }
        std::cout << "]\n";
    }
}