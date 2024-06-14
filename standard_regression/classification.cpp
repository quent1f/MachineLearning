#include "classification.h"

// Implementing logistic regression  or ReLU

double sigmoid(double x) {
    return 1/(1+exp(-x));
}

VectorXd sigmoid_vec(VectorXd &vec) {
    return 1/(1 + (-vec.array()).exp());
}

double reLU(double x) {
   return x < 0 ? 0 : x; 
}

VectorXd reLU_vec(VectorXd &vec) {
    return vec.cwiseMax(0); 
}


// Trying 1 layer neural network 

VectorXd predVector(MatrixXd &weights, VectorXd &bias, vector<double> &image) {
    VectorXd image_conv = Eigen::Map<VectorXd>(image.data(), image.size());
    VectorXd result = weights*image_conv + bias;
    return sigmoid_vec(result);
}

double getMaxIndex(VectorXd pred) {
    int max_index=0; 
    double max = pred[0];
    for (int i=0; i<pred.size(); i++) {
        if (pred[i] > max) {
            max = pred[i];
            max_index = i;
        }
    }
    return (double)max_index;
}

double prediction(MatrixXd &weights, VectorXd &bias, vector<double> &image) {
    return getMaxIndex(predVector(weights, bias, image));
}

double cost(VectorXd prediction, double result) {
    // Cost function is quadratic loss
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

double totalCost(MatrixXd &weights, VectorXd bias, vector<vector<double>> &data, vector<double> &labels) {
    int n = data.size();
    double res = 0;
    for (int i=0; i < n; i++) {
        VectorXd pred = predVector(weights, bias, data[i]);
        res += cost(pred, labels[i]);
    }
    return res;
}

    
// Training Neural Network : Backpropagation







// Testing Neural Network 

double hitRate(MatrixXd &weights, VectorXd bias, vector<vector<double>> &testSet, vector<double> &testLabels) {
    int n = testLabels.size();      // dataSet size
    int c = 0;                      // count the good predictions
    for (int i=0; i<n; i++) {
        if (prediction(weights, bias, testSet[i]) == testLabels[i]) {
            c++; 
        }
    }
    cout << "On the test set of size " << n << ", the model had " << c << "good answers. Accuracy : " << c/n << "\n";
    return c/n;
}   

