#pragma once

#include "Utils.hpp"
#include <vector>
#include <numeric>
#include "Functions.hpp"
#include "ProblemSpec.hpp"

#define Population std::vector<Bitstring>
#define Chromosome Bitstring
typedef Population (*SelectionStrategy)(Population&, std::vector<double>&, int);

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

std::pair<Bitstring, Bitstring> crossover(const Bitstring& parent1, const Bitstring& parent2) {
  std::uniform_int_distribution<> rand_chrom_len(1, parent1.size() - 2);
  int cutoff = rand_chrom_len(rng_generator);
  Bitstring child1(parent1.begin(), parent1.begin() + cutoff);
  Bitstring child2(parent2.begin(), parent2.begin() + cutoff);
  child1.insert(child1.end(), parent2.begin() + cutoff + 1, parent2.end());
  child2.insert(child2.end(), parent1.begin() + cutoff + 1, parent1.end());

  return {child1, child2};
}

Population crossoverPopulation(const Population& population) {
  Population children;
  std::shuffle(population.begin(), population.end(), rng_generator);
  for (int i = 0; i < population.size(); i += 2) {
    auto [child1, child2] = crossover(population[i], population[i + 1]);
    children.push_back(child1);
    children.push_back(child2);
  }
  return children;
}

inline void mutate(Bitstring& chromosome, double mutationRate) {
  for(int i = 0; i < chromosome.size(); ++i) {
    if (rand_real(rng_generator) < mutationRate) {
      chromosome[i] = !chromosome[i];
    }
  }
}

inline void mutatePopulation(Population& population, double mutationRate) {
  for (auto chrom : population) {
    mutate(chrom, mutationRate);
  }
}

/**
 * Select newPopSize individuals based on random 1v1 tournaments.
 */
Population tournamentSelection(Population& population, std::vector<double>& fitness, int newPopSize) {
  Population selected;
  std::uniform_int_distribution<> rand_chrom(0, population.size() - 1);
  for (int i = 0; i < newPopSize; ++i) {
    int idx1 = rand_chrom(rng_generator);
    int idx2 = rand_chrom(rng_generator);
    if (fitness[idx1] < fitness[idx2]) {
      selected.push_back(population[idx1]);
    } else {
      selected.push_back(population[idx2]);
    }
  }
  return selected;
}

// Population elitismSelection(Population& population, std::vector<double>& fitness, int newPopSize) {

// }

std::vector<int> sort_indexes(const std::vector<double>& populationFitness) {

  // initialize original index locations
  std::vector<int> idx(populationFitness.size());
  std::iota(idx.begin(), idx.end(), 0);
  std::stable_sort(idx.begin(), idx.end(),
       [&populationFitness](int i1, int i2) {return populationFitness[i1] < populationFitness[i2];});

  return idx;
}

Bitstring geneticAlgorithm(int popSize, int generations, float mutationRate, SelectionStrategy selection, const ProblemSpec& problem) {
  Population population = generateStartingPopulation(popSize, problem);
  std::vector<double> populationFitness = evaluatePopulation(population, problem);
  std::vector<int> sortedIndexes = sort_indexes(populationFitness);
  Chromosome bestSolution = population[sortedIndexes[0]];
  double bestScore = populationFitness[sortedIndexes[0]];
  for (int i = 0; i < generations; ++i) {
    // Selection
    population = selection(population, populationFitness, popSize);
    // Crossover
    Population children = crossoverPopulation(population);
    population.insert(population.end(), children.begin(), children.end());
    // Mutation
    std::for_each(population.begin(), population.end(), 
      [mutationRate](Chromosome& x){mutate(x, mutationRate);}
    );
    // Eval
    populationFitness = evaluatePopulation(population, problem);
    sortedIndexes = sort_indexes(populationFitness);
    Chromosome candidateSolution = population[sortedIndexes[0]];
    double candidateScore = populationFitness[sortedIndexes[0]];
    if (candidateScore < bestScore) {
      bestScore = candidateScore;
      bestSolution = candidateSolution;
    }
  }
  return bestSolution;
}