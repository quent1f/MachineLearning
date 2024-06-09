#include "classification.h"

// Implementing logistic regression 

double sigmoid(double x) {
    return 1/(1+exp(-x));
}