#include "softmax_layer.h"
#include <algorithm>


Softmax::Softmax()
{

}

Softmax::~Softmax()
{

}
void Softmax::Flatten_making(const std::vector<std::vector<double>>& img_input, int d)
{

   for (int i = 0; i<d ; i++){

           mFlatten.insert(mFlatten.end(), img_input[i].begin(), img_input[i].end());

   }

}

void Softmax::Cache_making()
{
    //Effacer le dernier cache et redimensionner
    mCachedFlatten.clear();
    mCachedTotal.clear();
    mCachedFlatten.resize(m_flatten.size());
    mCachedTotal.resize(m_total.size());

    //Copier la dernière entrée et les paramètres
    mCachedLength = m_length;
    mCachedFlatten.assign(m_flatten.begin(), m_flatten.end());
    mCachedTotal.assign(m_total.begin(), m_total.end());

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

    mFlatten.clear();//Effacer le dernier aplatissement de l'entrée
    mTotal.clear();//Effacer la derniere prédiction 


    Flatten_making(input_img, DEPTH);

    //Multiplier l'entrée aplatie et mWeights
    for (int i = 0; i < ND; i++) {
        double s = 0;
        // Boucle pour multiplier  mWeights[j] pour chaque  digit[i] avec chaque mFlatten [j]
        for (int j = 0; j < m_length; j++) {
            s += (mFlatten[j] * mWeights[j][i]);
        }
        //Somme de biais
        s += mBiases[i];
        mTotal.push_back(s);
    }
    std::vector<double> vect_exponentiel;
    std::vector<double> vect_predictions;

    double exp_s = 0.0;
    double t = 0.0;

    for (int i = 0; i < ND; i++)
    {
        t = exp(m_total[i]);
        vect_exponentiel.push_back(t);
        exp_s += (t);
    }

    for (int i = 0; i < ND; i++)
    {
        t = ((double)vect_exponentiel[i] / (double)exp_s);
        vect_predictions.push_back(t);
    }

    cache_making(); //Créer le cache de la dernière entrée

    return predictions;
}

}    



