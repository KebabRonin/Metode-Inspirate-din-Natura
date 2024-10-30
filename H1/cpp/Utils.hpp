#pragma once

#include <iostream>
#include <vector>
#include <random>

#define Bitstring std::vector<bool>
#define ParameterList std::vector<double>

const double PI = 3.14159265358979323846;

template < class T >
std::ostream& operator << (std::ostream& os, const std::vector<T>& v) 
{
    os << "[ ";
    for (auto i : v)
    {
        os << i << " ";
    }
    os << "]";
    return os;
}

std::random_device rd; // More random than pseudorandom maybe.
std::mt19937 rng_generator(1000000 + rd()); // Jump ahead so the RNG gets good.
std::uniform_int_distribution<> rand_binary(0, 1);
std::uniform_real_distribution<> rand_real;