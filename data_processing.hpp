#ifndef DATA_PROC
#define DATA_PROC

#include <iostream>
#include <vector>
#include <sstream>
#include <string>
#include <fstream>
#include <Eigen/Dense> 

using std::cout, std::vector, std::string;
using namespace Eigen;

VectorXd parseLine(string s);
    /* 
        Input : string s containing 28x28 integers in [0, 255] 
        Output : vector containing the normalised integers 
    */



vector<VectorXd> dataProcess(string f);
    /* 
        Input : string - file name of data 
        Output : Matrix of doubles in [0,1]. data[i] is a 28x28 1d array containing the information of 1 picture
    */


vector<double> labelProcess(string f);
    /*
        Input : string - file name of data labels
        Output : Vector labels where labels(i) is the integer represented by the image i 
    */



#endif