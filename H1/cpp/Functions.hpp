#pragma once

#include "Utils.hpp"

#include <iostream>
#include <cmath>
#include <vector>
#include <stdexcept>
#include <string>

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
	/**
	 * Return the value of the function when run with the given parameters.
	 * Also includes bound checking, so use `compute()` directly for more performance.
	 */
	double apply(const ParameterList& x) const {
		if (!check_bounds(x)) {
			std::cout << name() << ": x must be in [" << bounds().first << ", " << bounds().second << "]" << std::endl;
			exit(-1);
		}
		return compute(x);
	}
	virtual double compute(const Bitstring& x, const ProblemSpec& problem) const;
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
	new Rosenbrock()
	// ,new DeJong()
};
