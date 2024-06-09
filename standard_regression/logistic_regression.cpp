#include <iostream>
#include <vector>
#include <cmath>
#include <numeric>
#include <sstream>
#include <string>
#include <fstream>

using std::cout, std::vector, std::string;

// Implementation of logistic regression without bias

double sigmoid(double x) {
    return 1/(1+exp(-x));
}

double predictor(vector<double>& w, vector<double>& x) {
    double produit_scalaire = std::inner_product(w.begin(), w.end(), x.begin(), 0.0);
    return sigmoid(produit_scalaire);
}

void update_weights(vector<double>& w, vector<double>& x, double label, double lrate) {
    double prediction = predictor(w, x); 
    double error = prediction-label;
    for (int i=0; i < (int)w.size(); i++) {
        w[i] -= lrate*error*x[i];
    }
}

// Goal : Numbers classification 
// Input : 28*28 image with a written digit  
// OUtput : The digit 

// Processing data from a CSV file 

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

vector<int> labelProcess(string f) {
    /*
        Input : string - file name of data labels
        Output : Vector labels where labels[i] is the integer represented by the image i 
    */
    std::ifstream file(f); 
    string line; 
    vector<int> labels;
    while(std::getline(file, line)) {
        int x = std::stoi(line);
        labels.push_back(x);
    }
    file.close();
    return labels;
}

vector<double> train_model(double lrate, vector<vector<double>>& data, vector<double>& labels, vector<double>& w_init) {
    int n = data.size();
    vector<double> w = w_init;
    for (int i=0; i<n; i++) {
        update_weights(w, data[i], labels[i], lrate);
    }
    return w; 
}

double loss(double label, double prediction) {
    return 0;
}

double testPerf(vector<double>& w, vector<vector<double>>& data, vector<double>& labels) {
    int n = data.size();
    double fail = 0; 
    for (int i = 0; i<n; i++) {
        double prediction = 0;
        double prob = predictor(w, data[i]);
        if (prob > 0.2) {
            prediction = 1;
        }
        if (prediction == labels[i]) {
            // cout << i+1 << " prédiction réussie\n";
        }
        else {
            // cout << i+1 << " prédiction failed : " << " On a prédit : " << prediction << " et le résultat est : " << labels[i] << "\n";
        }
        fail += abs(prediction-labels[i]); 
    }
    cout << "Ratés : " << fail << " sur " << data.size() << "\n";
    cout << "Proba de fail : " << fail/data.size() << "\n";
    return fail;
}

void tests(vector<double>& w, vector<vector<double>>& data) {
    int n = data.size(); 
    for (int i=0; i<n; i++) {
        cout << i+1 << " "; 
        double prediction = predictor(w, data[i]); 
        cout << prediction << " "; 
        if (prediction > 0.01) {
            cout << 1 << "\n";
        }
        else {
            cout << 0 << "\n";
        }
    }
}


int main() {
    return 0;
}