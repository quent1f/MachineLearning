#include <iostream>
#include <vector>
#include <cmath>
#include <Eigen/Dense>
#include <random>
#include <sstream>
#include <string>
#include <fstream>

using std::cout, std::vector, std::string;
using namespace Eigen; 

// Processing Data and Labels

vector<double> parseLine(string s) {
    /* 
        Input : string s containing 28x28 integers in [0, 255] 
        Output : vector containing the normalised integers 
    */
    std::stringstream ss(s);
    string tmp;
    vector<double> res;
    while (getline(ss, tmp, ',')) {
        res.push_back(stoi(tmp) / 256.0);
    }
    return res;
}

vector<vector<double>> dataProcess(string f) {
    /* 
        Input : string - file name of data 
        Output : Matrix of doubles in [0,1]. data[i] is a 28x28 1d array containing the information of 1 picture
    */
    std::ifstream file(f);
    string line; 
    vector<vector<double>> data;
    while(std::getline(file, line)) {
        data.push_back(parseLine(line));
    }
    file.close();
    return data;
}

vector<double> labelProcess(string f) {
    /*
        Input : string - file name of data labels
        Output : Vector labels where labels[i] is the integer represented by the image i 
    */
    std::ifstream file(f); 
    string line; 
    vector<double> labels;
    while(std::getline(file, line)) {
        double x = std::stoi(line);
        labels.push_back(x);
    }
    file.close(); 
    return labels;
}

// Implementing logistic regression 

double sigmoid(double x) {
    return 1/(1+exp(-x));
}

VectorXd sigmoid_vec(VectorXd &vec) {
    int size = vec.size();
    VectorXd res(size);
    for (int i=0; i < size; i++) {
        res[i] = sigmoid(vec[i]);
    }
    return res;
}


VectorXd prediction(MatrixXd &weights, vector<double> &image) {
    VectorXd image_conv = Eigen::Map<VectorXd>(image.data(), image.size());
    VectorXd result = weights*image_conv;
    return result;
}

double cost(VectorXd prediction, double result) {
    double cost = 0;
    for (int i=0; i < 10; i++) {
        if (i == result) {
            cost += (prediction[i]-1)*(prediction[i]-1);
        }
        else {
            cost += prediction[i]*prediction[i];
        }
    }
    return cost;
}

double totalCost(MatrixXd &weights, vector<vector<double>> &data, vector<double> &labels) {
    int n = data.size();
    double res = 0;
    for (int i=0; i < n; i++) {
        VectorXd pred = prediction(weights, data[i]);
        res += cost(pred, labels[i]);
    }
    return res;
}


int main() {


    return 0; 
}