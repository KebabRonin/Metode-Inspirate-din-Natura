#include <cmath>
#include <iostream>
#include <random>
#include <vector>
#include <chrono>
#include <algorithm>
#include "functions.cpp"
#include "ProblemSpec.cpp"
#include "tqdm.hpp"
// #include "tqdm/tqdm.h"

#define Bitstring std::vector<bool>
#define ParameterList std::vector<double>


// Improves the given solution by evaluating neighbors and selecting the best one
Bitstring improve(Bitstring& vc, const ProblemSpec& problem) {
	double best_score = problem.f.compute(vc, problem);
	int best_neighbor = -1;
	int size = vc.size();

	for (int i = 0; i < size; ++i) {
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
	auto pbar = tq::trange(0, max_iter);
	pbar.set_prefix(problem.f.name());

	for (int t : pbar) {
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
	int n_dim = 5;
	int max_iter = 100;
	int precision = 5;
	std::cout << "Running Hill Climbing: " << "n_dim=" << n_dim << ", max_iter=" << max_iter << ", precision=" << precision << "\n";

	for (const OptimizationFunction* func_obj : FUNCTIONS) {
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