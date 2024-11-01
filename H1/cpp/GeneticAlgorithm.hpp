#pragma once

#include "Utils.hpp"
#include <vector>
#include <numeric>
#include "Functions.hpp"
#include "ProblemSpec.hpp"

// Genetic algorithm specific types
typedef std::vector<Bitstring> Population;
typedef Bitstring Chromosome;
typedef Population (*SelectionStrategy)(const Population&, const std::vector<double>&, int);


std::vector<int> sortIndexes(const std::vector<double>& populationFitness) {

  // initialize original index locations
  std::vector<int> idx(populationFitness.size());
  std::iota(idx.begin(), idx.end(), 0);
  std::sort(idx.begin(), idx.end(),
       [&populationFitness](int i1, int i2) {return populationFitness[i1] < populationFitness[i2];});

  return idx;
}

Population generateStartingPopulation(const int popSize, const ProblemSpec& problem) {
  Population pop;
  pop.reserve(popSize);

  for (int i = 0; i < popSize; ++i) {
    pop.push_back(problem.randomBitstring());
  }

  return pop;
}

std::vector<double> evaluatePopulation(const Population& pop, const ProblemSpec& problem) {
  std::vector<double> fitness;
  fitness.reserve(pop.size());
  for (const Chromosome chrom : pop) {
    fitness.push_back(problem.getFitness(chrom));
  }
  return fitness;
}

std::pair<Bitstring, Bitstring> crossover(const Bitstring& parent1, const Bitstring& parent2) {
  std::uniform_int_distribution<> rand_chrom_len(1, parent1.size() - 2);
  int cutoff = rand_chrom_len(rng_generator);
  Bitstring child1(parent1.begin(), parent1.begin() + cutoff);
  Bitstring child2(parent2.begin(), parent2.begin() + cutoff);
  child1.insert(child1.end(), parent2.begin() + cutoff, parent2.end());
  child2.insert(child2.end(), parent1.begin() + cutoff, parent1.end());

  return {child1, child2};
}

Population crossoverPopulation(const Population& population) {
  Population children;
  // Shuffle
  std::vector<double> randomValues;
  randomValues.reserve(population.size());
  std::for_each(population.begin(), population.end(), [&](const Chromosome& _) {randomValues.push_back(rand_real(rng_generator));});
  std::vector<int> sortedIndexes = sortIndexes(randomValues);
  for (auto i = sortedIndexes.begin(); i != sortedIndexes.end(); ++i, ++i) {
    auto j = i + 1;
    if (j == sortedIndexes.end()) {
      j = i; // sortedIndexes.begin();
    }
    auto [child1, child2] = crossover(population[*i], population[*j]);
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
Population tournamentSelection(const Population& population, const std::vector<double>& fitness, int newPopSize) {
  Population selected;
  selected.reserve(newPopSize);
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


Population elitismSelection(const Population& population, const std::vector<double>& fitness, int newPopSize) {
  std::vector<int> sortedIndexes = sortIndexes(fitness);
  Population newPopulation;
  newPopulation.reserve(newPopSize);
  std::for_each(sortedIndexes.begin(), sortedIndexes.begin() + newPopSize, [&](int idx) {newPopulation.push_back(population[idx]);});
  return newPopulation;
}


Population mixedSelection(const Population& population, const std::vector<double>& fitness, int newPopSize) {
  int elitismCount = newPopSize / 10; // 10% elitism
  int tournamentCount = newPopSize - elitismCount;
  Population selectedElitism = elitismSelection(population, fitness, elitismCount);
  Population selectedTournament = tournamentSelection(population, fitness, tournamentCount);

  selectedElitism.insert(selectedElitism.end(), selectedTournament.begin(), selectedTournament.end());
  return selectedElitism;
}

ParameterList geneticAlgorithm(int popSize, int generations, double mutationRate, SelectionStrategy selection, const ProblemSpec& problem) {
  Population population = generateStartingPopulation(popSize, problem);
  std::vector<double> populationFitness = evaluatePopulation(population, problem);
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
    populationFitness = evaluatePopulation(population, problem);
    sortedIndexes = sortIndexes(populationFitness);
    Chromosome candidateSolution = population[sortedIndexes[0]];
    double candidateScore = populationFitness[sortedIndexes[0]];
    if (candidateScore < bestScore) {
      bestScore = candidateScore;
      bestSolution = candidateSolution;
    }
    mutationRate *= 0.999;
  }
  ParameterList best_decoded = problem.decodeSolution(bestSolution);
  return best_decoded;
}