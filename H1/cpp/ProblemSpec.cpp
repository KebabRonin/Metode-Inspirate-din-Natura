#ifndef PROBLEM_SPEC_H
#define PROBLEM_SPEC_H

#include <iostream>
#include <cmath>
#include <vector>
#include <stdexcept>
#include <string>
#include <random>
#include <chrono>
#include <algorithm>
#include "functions.cpp"

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

#endif