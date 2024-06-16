#include "../data_processing.hpp"
#include "oneLayerSigmoid.hpp"

// Testing the NN 

void hitRate(MatrixXd &weights, VectorXd bias, vector<VectorXd> &testSet, vector<double> &testLabels) {
    int n = testLabels.size();      // dataSet size
    int c = 0;                      // count the good predictions
    for (int i=0; i<n; i++) {
        VectorXd pred = predVector(weights, bias, testSet[i]);
        int max_index = 0;
        double max = pred(0);  
        for (int j=1; j<10; j++) {
            if (max < pred(j)) {
                max_index = j; 
                max = pred(j);
            }
        }
        if (max_index == testLabels[i]) {
            c++; 
        }
    }
    cout << "On the test set of size " << n << ", the model had " << c << " good answers. Accuracy : " << (double)c/n << "\n";
    return; 
}   

void testingProcess(vector<VectorXd> trainSetImages, vector<double> trainSetLabels, vector<VectorXd> testSetImages, vector<double> testSetLabels) {

    MatrixXd weights = initWeights(1.0, 10, 784);
    VectorXd bias = initBias(1.0, 10);
    
    cout << "total cost before training : " << totalCost(weights, bias, trainSetImages, trainSetLabels) << "\n";

    trainModel1(weights, bias, trainSetImages, trainSetLabels, 0.2, 100, 10000);

    cout << "total cost after training : " << totalCost(weights, bias, testSetImages, testSetLabels) << "\n";
    hitRate(weights, bias, testSetImages, testSetLabels);


}

int main() {

    // Loading data set

    vector<VectorXd> trainSetImages = dataProcess("../data/x_train.csv");
    vector<double> trainSetLabels = labelProcess("../data/y_train.csv");

    vector<VectorXd> testSetImages = dataProcess("../data/x_test.csv");
    vector<double> testSetLabels = labelProcess("../data/y_test.csv");


    // cout << "Size of tranning dataset : " << trainSetLabels.size() <<  "\n";

    testingProcess(trainSetImages, trainSetLabels, testSetImages, testSetLabels);


    return 0; 
}