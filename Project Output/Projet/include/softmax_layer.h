#pragma once


#include "Random_weights.h"
#include <vector>

//Macros pour la couche softmax
#define ND 10
#define PROFD 8


class Softmax {
public:

	Softmax();   //constructeur
	~Softmax();  //destructeur

	std::vector<double> Softmax_start(const std::vector<std::vector<double>>& img_input, int img_height, int img_width);

	
private:

	void Flatten_making(const std::vector<std::vector<double>>& img_input, int d);

	int mWidth; 

	std::vector<std::vector<double>> m_weight;    // vecteur de poids 


	std::vector<double> m_biases;        // vecteur de biais

	
	
	

};

