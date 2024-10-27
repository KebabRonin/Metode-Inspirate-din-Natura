#ifndef FUNCTIONS_H
#define FUNCTIONS_H

#include <iostream>
#include <cmath>
#include <vector>
#include <stdexcept>
#include <string>

#define Bitstring std::vector<bool>
#define ParameterList std::vector<double>

const double PI = 3.14159265358979323846;

struct ProblemSpec;

class OptimizationFunction {
public:
	bool check_bounds(const ParameterList& x) const {
		for (double i : x) {
			if (i < bounds().first || i > bounds().second) {
				return false;
			}
		}
		return true;
	}
	double apply(const ParameterList& x) const {
		if (!check_bounds(x)) {
			std::cout << name() << ": x must be in [" << bounds().first << ", " << bounds().second << "]" << std::endl;
			exit(-1);
		}
		return compute(x);
	}
	double apply(const Bitstring& x, const ProblemSpec& problem) const;
	double compute(const Bitstring& x, const ProblemSpec& problem) const;
	virtual double compute(const ParameterList& x) const = 0;
	virtual std::pair<double, double> bounds() const = 0;
	virtual std::string name() const = 0;
};


class Michalewicz : public OptimizationFunction {
public:
	double compute(const ParameterList& x) const override {
		int n = x.size();
		int m = 10;
		double y = 0.0;
		for (int i = 0; i < n; ++i) {
			y -= std::sin(x[i]) * std::pow(std::sin(((i + 1) * std::pow(x[i], 2)) / PI), 2 * m);
		}
		return y;
	}

	std::pair<double, double> bounds() const override { return {0, PI}; }
	std::string name() const override { return "Michalewicz"; }
};


class Rastrigin : public OptimizationFunction {
public:
	double compute(const ParameterList& x) const override {
		int n = x.size();
		double y = 10 * n;
		for (double i : x) {
			y += std::pow(i, 2) - 10 * std::cos(2 * PI * i);
		}
		return y;
	}

	std::pair<double, double> bounds() const override { return {-5.12, 5.12}; }
	std::string name() const override { return "Rastrigin"; }
};


class Griewangk : public OptimizationFunction {
public:
	double compute(const ParameterList& x) const override {
		int n = x.size();
		double y_sum = 0.0;
		double y_prod = 1.0;
		for (int i = 0; i < n; ++i) {
			y_sum += std::pow(x[i], 2);
			y_prod *= std::cos(x[i] / std::sqrt(i + 1));
		}
		return y_sum / 4000 - y_prod + 1;
	}

	std::pair<double, double> bounds() const override { return {-600, 600}; }
	std::string name() const override { return "Griewangk"; }
};


class Rosenbrock : public OptimizationFunction {
public:
	double compute(const ParameterList& x) const override {
		int n = x.size();
		double y = 0.0;
		for (int i = 0; i < n - 1; ++i) {
			y += 100 * std::pow(x[i + 1] - std::pow(x[i], 2), 2) +
				std::pow((1 - x[i]), 2);
		}
		return y;
	}

	std::pair<double, double> bounds() const override { return {-2.048, 2.048}; }
	std::string name() const override { return "Rosenbrock"; }
};


class DeJong : public OptimizationFunction {
public:
	double compute(const ParameterList& x) const override {
		int n = x.size();
		double y = 0.0;
		for (int i = 0; i < n - 1; ++i) {
			y += std::pow(x[i], 2);
		}
		return y;
	}

	std::pair<double, double> bounds() const override { return {-5.12, 5.12}; }
	std::string name() const override { return "DeJong"; }
};


static const OptimizationFunction* FUNCTIONS[] = {
	new Michalewicz(),
	new Rastrigin(),
	new Griewangk(),
	new Rosenbrock(),
	new DeJong()
};

#endif
// int main() {
// 	ParameterList test_vec1 = {0, 0};
// 	ParameterList test_vec2 = {2.2, 1.57};
// 	ParameterList test_vec3 = {1, 1};

// 	ParameterList* current_vec;

// 	for (const auto& func : FUNCTIONS) {
// 		if (func->name() == "Michalewicz") {
// 			current_vec = &test_vec2;
// 		} else if (func->name() == "Rosenbrock") {
// 			current_vec = &test_vec3;
// 		} else {
// 			current_vec = &test_vec1;
// 		}
// 		std::cout << func->name() << " (test run): " << func->apply(*current_vec) << std::endl;
// 	}

// 	return 0;
// }