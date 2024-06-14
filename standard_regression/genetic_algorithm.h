#ifndef GENETIC
#define GENETIC 

#include <iostream>
#include <vector>
#include <cmath>
#include <Eigen/Dense>
#include <random>

using std::cout, std::vector, std::string;
using namespace Eigen;


MatrixXd generateRandomWeights(int rows, int cols);

VectorXd generateRandomBias(int rows);

MatrixXd testing(int n);







#endif