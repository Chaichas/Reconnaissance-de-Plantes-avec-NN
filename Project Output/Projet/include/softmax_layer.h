#pragma once


#include "Random_weights.h"
#include <vector>

//Macros pour la couche softmax
#define ND 10
#define DEPTH 8


class Softmax {
public:

	Softmax();   //constructeur
	~Softmax();  //destructeur

	std::vector<double> Softmax_start(const std::vector<std::vector<double>>& img_input, int img_height, int img_width);

	
private:


	int mLength; 

	std::vector<std::vector<double>> mWeight;    // vecteur de poids 


	


};

