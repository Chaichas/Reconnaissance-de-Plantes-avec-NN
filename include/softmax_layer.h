#pragma once


#include "Convolution_layer.h"
#include "Pooling_layer.h"
#include <vector>

//Macros pour la couche softmax
#define ND 5
#define DEPTH 8


class Softmax_layer {
public:

	Softmax_layer(){};   //constructeur
	~Softmax_layer(){};  //destructeur

	std::vector<double> Softmax_start(const std::vector<std::vector<double>>& img_input, int img_height, int img_width);
	//Backpropagation
	std::vector<std::vector<double>> BackPropagation(const std::vector<double>& d_L_d_out, const double learn_rate);
private:

	void Flatten(const std::vector<std::vector<double>>& img_input, int d);

	int mLength;
	bool  pred = true;

	std::vector<std::vector<double>> mWeights;    // vecteur de poids 

	std::vector<double> mBiases; 	//couche vecteur de biais 

	std::vector<double> mFlatten; //Vecteur pour stocker l'entrée en tant que vecteur d'aplatissement

	std::vector<double> mTotal; //Vecteur de stockage des pr�dictions finales pour chaque noeud (chiffre)

	void Hidden();//Fonction pour faire des caches de la derniére entrée


	int mCachedLength;
	std::vector<double> mCachedFlatten;
	std::vector<double> mCachedTotal;





};
