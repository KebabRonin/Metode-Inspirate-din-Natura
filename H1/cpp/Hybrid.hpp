#pragma once
#include "tqdm.hpp"
#include "Utils.hpp"
#include "GeneticAlgorithm.hpp"
#include "HillClimbing.hpp"

// pop is a copy so hc does not alter the population
std::vector<double> evaluatePopulationHybrid(Population& pop, const ProblemSpec& problem, int max_iter=100) {
  std::vector<double> fitness;
  fitness.reserve(pop.size());
  for (Chromosome chrom : pop) {
    fitness.push_back(hillClimbing(chrom, problem, max_iter).second);
  }
  return fitness;
}

ParameterList hybrid(
  int popSize, int generations, float mutationRate, SelectionStrategy selection,
  const ProblemSpec& problem, int hcIters=100
) {
  Population population = generateStartingPopulation(popSize, problem);
  std::vector<double> populationFitness = evaluatePopulationHybrid(population, problem, hcIters);
  std::vector<int> sortedIndexes = sortIndexes(populationFitness);
  Chromosome bestSolution = population[sortedIndexes[0]];
  double bestScore = populationFitness[sortedIndexes[0]];
  for (auto i : tq::trange(generations)) {
    // Selection
    population = selection(population, populationFitness, popSize / 2);
    // Crossover
    Population children = crossoverPopulation(population);
    population.insert(population.end(), children.begin(), children.end());
    // Mutation
    std::for_each(population.begin(), population.end(),
      [mutationRate](Chromosome& x){mutate(x, mutationRate);}
    );
    // Eval
    std::cout<<"hybrid population"<<&population<<std::endl;
    std::cout<<"hybrid population"<<&(population[0])<<std::endl;
    populationFitness = evaluatePopulationHybrid(population, problem, hcIters);
    sortedIndexes = sortIndexes(populationFitness);
    Chromosome candidateSolution = population[sortedIndexes[0]];
    double candidateScore = populationFitness[sortedIndexes[0]];
    if (candidateScore < bestScore) {
      bestScore = candidateScore;
      bestSolution = candidateSolution;
    }
    mutationRate *= 0.999;
  }
  bestSolution = hillClimbing(bestSolution, problem).first; // Added this
  ParameterList best_decoded = problem.decodeSolution(bestSolution);
  return best_decoded;
}