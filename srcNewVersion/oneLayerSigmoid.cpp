#include "oneLayerSigmoid.hpp"

// Useful functions 

VectorXd sigmoidVec(VectorXd &vec) {
    return 1/(1 + (-vec.array()).exp());
}

double sigmoid_prime(double x) {
    double exp_x = exp(-x);
    return exp_x/((1+exp_x)*(1+exp_x));
}

// Step 1 : Initializig NN 

MatrixXd initWeights(double maxWeight, int rows, int cols) {
    std::random_device rd;  
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-maxWeight, maxWeight);
    MatrixXd weights(rows, cols);
    for (int i=0; i<rows; i++) {
        for (int j=0; j<cols; j++) {
            weights(i,j) = dis(gen);
        }
    }
    return weights;
}

VectorXd initBias(double maxBias, int rows) {
    std::random_device rd;  
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(-maxBias, maxBias);
    VectorXd bias(rows);
    for (int i=0; i<rows; i++) {
        bias(i) = dis(gen);
    }
    return bias;
}

// Step 2 : Stochastic Gradient Descent to train the model

void updateWeightsAndBias(MatrixXd &weights, VectorXd &bias, double lrate, vector<VectorXd> &images, vector<double> labels, int s) {
    // int s represent the size of the batch used to compute the gradient (huge impact on performance)
    int dataSize = labels.size();
    std::random_device rd;  
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, dataSize-1);
    for (int i=0; i<s; i++) {
        int rd_int = dis(gen);
        VectorXd z = weights*images[rd_int] + bias;              // compute z = Wa + b where W are weights, a the input, b bias 
        VectorXd temp = sigmoidVec(z);                           // compute scalar product : <sigmoid(z), y> where y is the vector with the right prediction (0 everywhere except 1 in the right digit)
        int layer_size = z.size();                               // should be 10 in our exemple
        for (int j=0; j<layer_size; j++) {
            double temp2;
            if (j == labels[rd_int]) {
                temp2 = (lrate*sigmoid_prime(z(j))*(temp(j)-1))/s;
            }
            else {
                temp2 = (lrate*sigmoid_prime(z(j))*temp(j))/s;
            }
            bias(j) -= temp2;                                    // update bias(j)
            for (int k=0; k<784; k++) {
                weights(j,k) -= images[rd_int](k)*temp2;         // update weights(j,k)
            }
        }
    }
    return; 
} 


void trainModel1(MatrixXd &weights, VectorXd &bias, vector<VectorXd> &trainImages, vector<double> &trainLabels, double lrate, int s, int Nit) {        // Nit : number of iteration 
    // Iterating Gradient Descent 
    for (int i=0; i<Nit; i++) {
        updateWeightsAndBias(weights, bias, lrate, trainImages, trainLabels, s);
        // cout << "iteration : " <<  i << "\n";
    }
}

// The convergence is not that good with uniform learning rate. Let's decrease the lrate with number of iteration so we make big steps early and small steps later

void trainModel2(MatrixXd &weights, VectorXd &bias, vector<VectorXd> &trainImages, vector<double> &trainLabels, double lrate, int s, int Nit) {        // epsilon : iterating until avg_cost < epsilon
    // Iterating Gradient Descent 
    for (int i=0; i<Nit; i++) {
        updateWeightsAndBias(weights, bias, lrate, trainImages, trainLabels, s); 
        lrate *= 0.999;
    }
}

// Step 3 : using NN on new data and comparing performance

VectorXd predVector(MatrixXd &weights, VectorXd &bias, VectorXd &image) {
    VectorXd res = weights*image + bias;
    return sigmoidVec(res);
}

VectorXd labelToVec(double label) {
    VectorXd result = VectorXd::Zero(10); 
    result[(int)label] = 1;
    return result;
}

double totalCost(MatrixXd &weights, VectorXd bias, vector<VectorXd> &data, vector<double> &labels) {
    int n = data.size();
    double res = 0;
    for (int i=0; i < n; i++) {
        VectorXd pred = predVector(weights, bias, data[i]);
        res += (pred-labelToVec(labels[i])).squaredNorm();
    }
    return res/n;
}
