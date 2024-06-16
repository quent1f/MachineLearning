#include "data_processing.hpp"

// Processing data from a CSV file 

VectorXd parseLine(string s) {
    /* 
        Input : string s containing 28x28 integers in [0, 255] 
        Output : vector containing the normalised integers 
    */
    std::stringstream ss(s);
    string tmp;
    VectorXd res(28*28);
    int i = 0;
    while (getline(ss, tmp, ',')) {
        res(i) = stoi(tmp) / 256.0;
        i++;
    }
    return res;
}

vector<VectorXd> dataProcess(string f) {
    /* 
        Input : string - file name of data 
        Output : Matrix of doubles in [0,1]. data[i] is a 28x28 1d array containing the information of 1 picture
    */
    std::ifstream file(f);
    string line; 
    vector<VectorXd> data;
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



