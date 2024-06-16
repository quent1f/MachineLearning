#include "data_processing.hpp"
#include "classification.hpp"

int main() {

    // Loading data set
    vector<vector<double>> trainSetImages = dataProcess("data/x_train.csv");
    vector<double> trainSetLabels = labelProcess("data/y_train.csv");

    cout << "Size of tranning dataset : " << trainSetLabels.size() <<  "\n";

    // Training the model

    MatrixXd weights = initWeights(1.0, 10, 784);
    VectorXd bias = initBias(1.0, 10);

    trainModel(weights, bias, trainSetImages, trainSetLabels, 0.05, 500, 10000);


    // Testing the model 

    vector<vector<double>> testSetImages = dataProcess("data/x_test.csv");
    vector<double> testSetLabels = labelProcess("data/y_test.csv");

    hitRate(weights, bias, testSetImages, testSetLabels);

    return 0; 
}