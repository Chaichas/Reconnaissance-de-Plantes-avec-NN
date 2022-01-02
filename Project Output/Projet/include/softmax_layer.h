#pragma once


#include "Random_weights.h"
#include <vector>

//Macros pour la couche softmax
#define ND 5
#define DEPTH 8





class Softmax {
public:

	Softmax();   //constructeur
	~Softmax();  //destructeur

	std::vector<double> Softmax_start(const std::vector<std::vector<double>>& img_input, int img_height, int img_width);
    
	
private:
    
	void Flatten_making(const std::vector<std::vector<double>>& img_input, int d);
    
	int mLength; 

	std::vector<std::vector<double>> mWeights;    // vecteur de poids 
	
	std::vector<double> mBiases; 	//couche vecteur de biais 

	std::vector<double> mFlatten; //Vecteur pour stocker l'entrée en tant que vecteur d'aplatissement
	
	std::vector<double> mTotal; //Vecteur de stockage des prédictions finales pour chaque nœud (chiffre)

	void Cache_making();//Fonction pour faire des caches de la dernière entrée

    int mCachedLength;
	std::vector<double> mCachedFlatten;
	std::vector<double> mCachedTotal;


	


};

