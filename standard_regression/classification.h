#ifndef CLASSIFICATION 
#define CLASSIFICATION 

#include <iostream>
#include <vector>
#include <cmath>
#include <Eigen/Dense>

using std::cout, std::vector, std::string;
using namespace Eigen;
 
// useful functions 

double sigmoid(double x); // sigmoid function

VectorXd sigmoid_vec(VectorXd &vec); // applying sigmoid function to every component of the vector 

double reLU(double x);

VectorXd reLU_vec(VectorXd &vec);

VectorXd predVector(MatrixXd &weights, VectorXd &bias, vector<double> &image);

double getMaxIndex(VectorXd pred);

double prediction(MatrixXd &weights, VectorXd &bias, vector<double> &image);

double cost(VectorXd prediction, double result);

double totalCost(MatrixXd &weights, vector<vector<double>> &data, vector<double> &label);

// training NN 



// testing NN 

double hitRate(MatrixXd &weights, VectorXd bias, vector<vector<double>> &testSet, vector<double> &testLabels);

#endif 