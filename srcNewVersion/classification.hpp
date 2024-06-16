#ifndef CLASSIFICATION 
#define CLASSIFICATION 

#include <iostream>
#include <vector>
#include <cmath>
#include <random> 
#include <Eigen/Dense>

using std::cout, std::vector, std::string;
using namespace Eigen;
 
// useful functions 

double sigmoid(double x); // sigmoid function

VectorXd sigmoidVec(VectorXd &vec); // applying sigmoid function to every component of the vector 

VectorXd sigmoidVec2(VectorXd &vec);

double sigmoid_prime(double x); // sigmoid derivative 

double reLU(double x);

VectorXd reLU_vec(VectorXd &vec);

double heaviside(double x);

VectorXd predVector(MatrixXd &weights, VectorXd &bias, vector<double> &image);

double getMaxIndex(VectorXd pred);

double prediction(MatrixXd &weights, VectorXd &bias, vector<double> &image);

double cost(VectorXd prediction, double result);

double totalCost(MatrixXd &weights, VectorXd bias, vector<vector<double>> &data, vector<double> &labels);

// training NN 

MatrixXd initWeights(double maxWeight, int rows, int cols);

VectorXd initBias(double maxBias, int rows);

void updateWeightsAndBias(MatrixXd &weights, VectorXd &bias, double lrate, vector<VectorXd> &images, vector<double> labels, int s);

void trainModel(MatrixXd &weights, VectorXd &bias, vector<vector<double>> &trainImages, vector<double> &trainLabels, double lrate, int s, int Nit);



// testing NN 

void hitRate(MatrixXd &weights, VectorXd bias, vector<vector<double>> &testSet, vector<double> &testLabels);

#endif 