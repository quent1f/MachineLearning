#ifndef ONELAYERSIGM 
#define ONELAYERSIGM 

#include <iostream>
#include <vector>
#include <cmath>
#include <random> 
#include <Eigen/Dense>

using std::cout, std::vector, std::string;
using namespace Eigen;

// Useful stuff

VectorXd sigmoidVec(VectorXd &vec);

double sigmoid_prime(double x);

VectorXd sigmoid_primeVec(const VectorXd &vec);

// Init NN 

MatrixXd initWeights(double maxWeight, int rows, int cols);

VectorXd initBias(double maxBias, int rows);

// Training NN 

void updateWeightsAndBias(MatrixXd &weights, VectorXd &bias, double lrate, vector<VectorXd> &images, vector<double> labels, int s);

void trainModel1(MatrixXd &weights, VectorXd &bias, vector<VectorXd> &trainImages, vector<double> &trainLabels, double lrate, int s, int Nit);

void trainModel2(MatrixXd &weights, VectorXd &bias, vector<VectorXd> &trainImages, vector<double> &trainLabels, double lrate, int s, int Nit);

// Testing NN 

VectorXd predVector(MatrixXd &weights, VectorXd &bias, VectorXd &image);

VectorXd labelToVec(double label);

double totalCost(MatrixXd &weights, VectorXd bias, vector<VectorXd> &data, vector<double> &labels);


#endif