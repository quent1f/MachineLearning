#include "classification.h"

// Implementing logistic regression 

double sigmoid(double x) {
    return 1/(1+exp(-x));
}

VectorXd sigmoid_vec(VectorXd &vec) {
    int size = vec.size();
    VectorXd res(size);
    for (int i=0; i < size; i++) {
        res[i] = 1/(1+exp(-(vec[i])));
    }
    return res;
}

// Trying 1 layer neural network 

VectorXd prediction(MatrixXd &weights, VectorXd bias, vector<double> &image) {
    VectorXd image_conv = Eigen::Map<VectorXd>(image.data(), image.size());
    VectorXd result = weights*image_conv + bias;
    return sigmoid_vec(result);
}

double cost(VectorXd prediction, double result) {
    double cost = 0;
    for (int i=0; i < 10; i++) {
        if (i == result) {
            cost += (prediction[i]-1)*(prediction[i]-1);
        }
        else {
            cost += prediction[i]*prediction[i];
        }
    }
    return cost;
}

double totalCost(MatrixXd &weights, vector<vector<double>> &data, vector<double> &labels) {
    int n = data.size();
    double res = 0;
    for (int i=0; i < n; i++) {
        VectorXd pred = prediction(weights, data[i]);
        res += cost(pred, labels[i]);
    }
    return res;
}






// Prochaine étape : définir update_weights : comment on actualise les poids ? 

/* 
Réseau de neurones à 1 couche : entrée image -> sortie chiffre entre 0 et 9 
On a 784*10 + 10 paramètre 
Ne pas oublier le biais
sigmoid(WX + b)

faire une fonction qui s'entraine à partir des données 
faire une fonction qui teste 
Affichage meilleur 


Idée : algo génétique pour trouver minimiser le cout 

- generer des weights aléatoires 
- mesurer le cout 
- garder les meilleurs 
- merge le tout 
- réiterer 

*/