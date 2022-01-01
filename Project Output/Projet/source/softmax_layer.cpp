#include "softmax_layer.h"
#include <algorithm>


Softmax::Softmax()
{

}

Softmax::~Softmax()
{

}

std::vector<double> Softmax::Softmax_start(const std::vector<std::vector<double>>& img_input, int img_height, int img_width)
{
    m_lenght  = img_height * img_width * DEPTH;
    bool  init = true;

    //Pour l'initialisation des poids et des biais en première exécution
    if (init)
    {
        //Attribuez 0 à m_biases (la longueur de m_biase est de 10)
        m_biases.assign(ND, 0.0);

        //Distribution de nombres aléatoires qui produit des valeurs à virgule flottante selon une distribution normale
        Random_weights(mLength, ND, mWeights);
        

    
        size_t  i = 0 , j = 0;      
        while(i< mLength){
             while(i< mLength){
                  mWeights[i][j] = (double)mWeights[i][j] / (double)mLength;
                  j++;
             }
             i++;
        }
       
        init = false;
    }
}    