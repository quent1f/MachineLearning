#include "genetic_algorithm.h"
#include "classification.h"
#include "data_processing.h"

// Generating weights and bias

MatrixXd generateRandomWeights(int rows, int cols) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-1.0, 1.0);
    MatrixXd weights(rows, cols);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            weights(i, j) = dis(gen);
        }
    }
    return weights;
}

VectorXd generateRandomBias(int rows) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-1.0, 1.0);
    VectorXd bias(rows);
    for (int i = 0; i < rows; ++i) {
        bias(i) = dis(gen);
    }
    return bias;
}

// Oriented Object Programming ? 

MatrixXd testing(int n) {
    MatrixXd best_weights = generateRandomWeights(10, 784);
    vector<vector<double>> data = dataProcess("data/x_train.csv");
    vector<double> labels = labelProcess("data/y_train.csv");
    double best_cost = totalCost(best_weights, data, labels);
    for (int i = 0; i<n; i++) {
        MatrixXd weights = generateRandomWeights(10, 784);
        double cost = totalCost(weights, data, labels);
        if (cost > best_cost) {
            best_weights = weights;
            best_cost = cost; 
        }
        cout << i << "iteration \n"; 
    }
    cout << "best_cost : " << best_cost << "\n";
    return best_weights;
}   