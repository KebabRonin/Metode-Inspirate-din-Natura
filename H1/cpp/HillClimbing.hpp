#pragma once

#include <cmath>
#include <random>
#include <chrono>
#include <algorithm>

#include "Utils.hpp"
#include "Functions.hpp"
#include "ProblemSpec.hpp"
#include "tqdm.hpp"

/**
 * Improves the given bitstring by evaluating all neighbours (best fit).
 * Modifies the given bitstring on return.
 */
void improve(Bitstring& vc, const ProblemSpec& problem) {
	double best_score = problem.getFitness(vc);
	int best_neighbor = -1;
	int size = vc.size();

	for (int i = 0; i < size; ++i) {
		vc[i] = !vc[i];
		double neighbor_score = problem.getFitness(vc);
		if (neighbor_score < best_score) {
			best_score = neighbor_score;
			best_neighbor = i;
		}
		vc[i] = !vc[i];
	}

	if (best_neighbor != -1) {
		vc[best_neighbor] = !vc[best_neighbor];
	}
}


std::pair<Bitstring&, double> hillClimbing(Bitstring& vc, const ProblemSpec& problem, int max_iter=10) {
	double vc_score = problem.getFitness(vc);
	bool local = false;

	while (!local && max_iter--) {
		improve(vc, problem); // vc is modified here if a better neighbour is found.
		double vn_score = problem.getFitness(vc);

		if (vn_score < vc_score) {
			// vc = vn;
			vc_score = vn_score;
		} else {
			local = true;
		}
	}

	return {vc, vc_score};
}


std::pair<ParameterList, double> iteratedHillClimbing(const int max_iter, const ProblemSpec& problem) {
	// Initial guess
	Bitstring best = problem.randomBitstring();
	double best_score = problem.getFitness(best);

	// Logging time taken and progress bar.
	auto pbar = tq::trange(0, max_iter);
	pbar.set_prefix(problem.f.name());

	for (int t : pbar) {
		Bitstring vc = problem.randomBitstring();

		auto [vn, vn_score] = hillClimbing(vc, problem);

		if (vn_score < best_score) {
			best = vn;
			best_score = vn_score;
		}
	}

	ParameterList best_decoded = problem.decodeSolution(best);
	return {best_decoded, best_score};
}