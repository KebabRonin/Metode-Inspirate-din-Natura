#pragma once

#include "Utils.hpp"
#include <vector>
#include "Functions.hpp"
#include "ProblemSpec.hpp"

#define Population std::vector<Bitstring>
#define Chromosome Bitstring

Population generateStartingPopulation(const int popSize, const ProblemSpec& problem) {
  Population pop(popSize);

  for (int i = 0; i < popSize; ++i) {
    pop.push_back(problem.randomBitstring());
  }

  return pop;
}

std::vector<double> evaluatePopulation(const Population& pop, const ProblemSpec& problem) {
  std::vector<double> fitness(pop.size());
  for (Chromosome chrom : pop) {
    fitness.push_back(problem.getFitness(chrom));
  }
  return fitness;
}

std::pair<Bitstring, Bitstring> crossover(const Bitstring& parent1, const Bitstring& parent2, const ProblemSpec& problem) {
  //TODO: Implement
  return {parent1, parent2};
}