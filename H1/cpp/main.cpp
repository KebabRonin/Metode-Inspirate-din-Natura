#include "Functions.hpp"
#include "ProblemSpec.hpp"
#include "HillClimbing.hpp"
#include "GeneticAlgorithm.hpp"

void runHC(int n_dim, int precision, int max_iter);
void runAG(int n_dim, int precision, int pop_size, int max_generations);
void printFunctions();

int main(int argc, char** argv) {
	int n_dim = 5;
	int max_iter = 100;
	int precision = 5;
	//printFunctions();
	// runHC(n_dim, precision, max_iter);
	runAG(n_dim, precision, 1000, 300);
}


void runAG(int n_dim, int precision, int pop_size, int max_generations) {
	std::cout << "{\n\t\"algorithm\" : {\n\t\t\"name\" : \"Genetic\",\n\t\t\"pop_size\" : " << 
			pop_size << ",\n\t\t\"max_generations\" : " << 
			max_generations << ",\n\t\t\"precision\" : " <<
			precision << "\n\t},";

	for (const OptimizationFunction* func_obj : FUNCTIONS) {
		auto start = std::chrono::high_resolution_clock::now();

		// Actual code
		ProblemSpec problem(*func_obj, precision, n_dim);
		double mutationRate = 1.0 / problem.sol_len;
		Bitstring best_solution = 
				geneticAlgorithm(pop_size, max_generations, mutationRate, 
				tournamentSelection, problem);
		double best_score = problem.getFitness(best_solution);

		// Printing
		auto end = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double> elapsed = end - start;
		std::cout << "\n\t\"" <<
				problem.f.name() << "\":{\n\t\t\"seconds_elapsed\" : " <<
				elapsed.count() << ",\n\t\t\"best_score\" : " <<
				best_score << ",\n\t\t\"best_solution\" : " <<
				best_solution << "\n\t\t},";
	}
	std::cout << "\n}";
}


void runHC(int n_dim, int precision, int max_iter)
{
	std::cout << "{\n\t\"algorithm\" : {\n\t\t\"name\" : \"Hill Climbing\",\n\t\t\"n_dim\" : " << 
			n_dim << ",\n\t\t\"max_iter\" : " << 
			max_iter << ",\n\t\t\"precision\" : " << precision << "\n\t},";

	for (const OptimizationFunction* func_obj : FUNCTIONS) {
		auto start = std::chrono::high_resolution_clock::now();

		// Actual code
		ProblemSpec problem(*func_obj, precision, n_dim);
		auto [best_solution, best_score] = iteratedHillClimbing(max_iter, problem);

		// Printing
		auto end = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double> elapsed = end - start;
		std::cout << "\n\t\"" <<
				problem.f.name() << "\":{\n\t\t\"seconds_elapsed\" : " <<
				elapsed.count() << ",\n\t\t\"best_score\" : " <<
				best_score << ",\n\t\t\"best_solution\" : " <<
				best_solution << "\n\t\t},";
	}
	std::cout << "\n}";
}


/**
 * Print the values of the functions in their optimum for 2 dimensions.
 */
void printFunctions() {
	ParameterList test_vec1 = {0, 0};
	ParameterList test_vec2 = {2.2, 1.57};
	ParameterList test_vec3 = {1, 1};

	ParameterList* current_vec;

	for (const auto& func : FUNCTIONS) {
		if (func->name() == "Michalewicz") {
			current_vec = &test_vec2;
		} else if (func->name() == "Rosenbrock") {
			current_vec = &test_vec3;
		} else {
			current_vec = &test_vec1;
		}
		std::cout << func->name() << " (test run): Value is " << func->apply(*current_vec) << " in " << *current_vec << std::endl;
	}
}
