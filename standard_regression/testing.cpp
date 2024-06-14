#include "testing.h"

double getMaxIndex(VectorXd &vec) {
    int max_index = 0;
    for (int i=0; i<vec.size(); i++) {
        if (vec[i] > vec[max_index]) {
            max_index = i;
        }
    } 
    return (double)max_index;
}

double hitRate(MatrixXd &weights, VectorXd &bias, vector<vector<double>> &testSet, vector<double> &testLabels) {
    int n = testLabels.size();
    for (int i=0; i<n; i++) {  
        VectorXd pred = prediction(weights, bias, testSet[i]);
        
    }

}
