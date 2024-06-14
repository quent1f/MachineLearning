#include "data_processing.h"
#include "classification.h"

int main() {

    // Loading data set
    vector<vector<double>> trainSetImages = dataProcess("data/x_train.csv");
    vector<double> trainSetLabels = labelProcess("data/y_train.csv");

    cout << "Size of tranning dataset : " << trainSetLabels.size() <<  "\n";

    // Training the model

    // MatrixXd weights = trainModel(trainSetImages, trainSetLabels);





    // Testing the model 

    vector<vector<double>> testSetImages = dataProcess("data/x_test.csv");
    vector<double> testSetLabels = labelProcess("data/y_test.csv");

    cout << "Size of testing dataset : " << testSetLabels.size() << "\n";

    // double rate = hitRate(weights, bias, testSetImages, testSetLabels);

    return 0; 
}