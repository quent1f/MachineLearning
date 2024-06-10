#ifndef CLASSIFICATION 
#define CLASSIFICATION 

#include <iostream>
#include <vector>
#include <cmath>
#include <Eigen/Dense>

using std::cout, std::vector, std::string;
using namespace Eigen;

double sigmoid(double x); // sigmoid function

VectorXd sigmoid_vec(VectorXd &vec); // applying sigmoid function to every component of the vector 

VectorXd prediction(MatrixXd &weights, vector<double> &image);

double cost(VectorXd prediction, double result);







#endif 