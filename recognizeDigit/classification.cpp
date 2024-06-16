#include "classification.hpp"

// Implementing logistic regression  or ReLU

double sigmoid(double x) {
    return 1/(1+exp(-x));
}

VectorXd sigmoidVec(VectorXd &vec) {
    return 1/(1 + (-vec.array()).exp());
}

double sigmoid_prime(double x) {
    double exp_x = exp(-x);
    return exp_x/((1+exp_x)*(1+exp_x));
}

VectorXd sigmoid_prime_vec(VectorXd &vec) {
    VectorXd sigmoid_vec = sigmoidVec(vec);
    return sigmoid_vec.array()*(1 - sigmoid_vec.array());
}

double reLU(double x) {
   return x < 0 ? 0 : x; 
}

VectorXd reLU_vec(VectorXd &vec) {
    return vec.cwiseMax(0); 
}

double heaviside(double x ) {
    return x < 0 ? 0 : 1; 
}


// Trying 1 layer neural network 

VectorXd predVector(MatrixXd &weights, VectorXd &bias, vector<double> &image) {
    VectorXd image_conv = Map<VectorXd>(image.data(), image.size());
    VectorXd result = weights*image_conv + bias;
    return sigmoidVec(result);
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

VectorXd labelToVec(double label) {
    VectorXd result = VectorXd::Zero(10); 
    result[(int)label] = 1;
    return result;
}

double prediction(MatrixXd &weights, VectorXd &bias, vector<double> &image) {
    return getMaxIndex(predVector(weights, bias, image));
}

double cost(VectorXd pred, double result) {
    // Cost function is quadratic loss
    return (pred-labelToVec(result)).squaredNorm();
}

double totalCost(MatrixXd &weights, VectorXd bias, vector<vector<double>> &data, vector<double> &labels) {
    int n = data.size();
    double res = 0;
    for (int i=0; i < n; i++) {
        VectorXd pred = predVector(weights, bias, data[i]);
        res += cost(pred, labels[i]);
    }
    return res/n;
}


// Training Neural Network : Stochastic Gradient Descent

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
                temp2 = (2*lrate*sigmoid_prime(z(j))*(temp(j)-1))/s;
            }
            else {
                temp2 = (2*lrate*sigmoid_prime(z(j))*temp(j))/s;
            }
            bias(j) -= temp2;                                    // update bias(j)
            for (int k=0; k<784; k++) {
                weights(j,k) -= images[rd_int](k)*temp2;         // update weights(j,k)
            }
        }
    }
    return; 
} 


void trainModel(MatrixXd &weights, VectorXd &bias, vector<vector<double>> &trainImages, vector<double> &trainLabels, double lrate, int s, int Nit) {        // Nit : number of iteration 
    int dataSize = trainLabels.size(); 
    vector<VectorXd> images(dataSize);                                                               // Switching to Eigen librairy for fast computing
    for (int i=0; i<dataSize; i++) {                                                                // TO OPTIMIZE (&images in MatrixXd ?)
        images[i] = Map<VectorXd>(trainImages[i].data(), trainImages[i].size());
    }
    for (int i=0; i<Nit; i++) {
        updateWeightsAndBias(weights, bias, lrate, images, trainLabels, s);
        // cout << "iteration : " <<  i << "\n";
    }
}

// Testing Neural Network 

void hitRate(MatrixXd &weights, VectorXd bias, vector<vector<double>> &testSet, vector<double> &testLabels) {
    int n = testLabels.size();      // dataSet size
    int c = 0;                      // count the good predictions
    for (int i=0; i<n; i++) {
        if (prediction(weights, bias, testSet[i]) == testLabels[i]) {
            c++; 
        }
    }
    cout << "On the test set of size " << n << ", the model had " << c << " good answers. Accuracy : " << (double)c/n << "\n";
    return; 
}   

