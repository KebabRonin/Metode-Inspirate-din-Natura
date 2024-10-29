#pragma once

#include <iostream>
#include <vector>

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