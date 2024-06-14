#ifndef TESTING 
#define TESTING 

#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include "data_processing.h"
#include "classification.h"

using namespace Eigen;

double getMaxIndex(VectorXd &vec);

double hitRate(MatrixXd &weights, VectorXd &bias, vector<vector<double>> &testSet, vector<double> &testLabels);







#endif 