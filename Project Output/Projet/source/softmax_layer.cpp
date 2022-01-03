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
    mCachedFlatten.resize( mFlatten.size());
    mCachedTotal.resize( mTotal.size());

    //Copier la dernière entrée et les paramètres
    mCachedLength =  mLength;
    mCachedFlatten.assign( mFlatten.begin(),  mFlatten.end());
    mCachedTotal.assign( mTotal.begin(),  mTotal.end());

}



std::vector<double> Softmax::Softmax_start(const std::vector<std::vector<double>>& img_input, int img_height, int img_width)
{
     mLength  = img_height * img_width * DEPTH;
    bool  init = true;

    //Pour l'initialisation des poids et des biais en première exécution
    if (init)
    {
        //Attribuez 0 à  mBiases (la longueur de m_biase est de 10)
         mBiases.assign(ND, 0.0);

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
        for (int j = 0; j <  mLength; j++) {
            s += (mFlatten[j] * mWeights[j][i]);
        }
        //Somme de biais
        s += mBiases[i];
        mTotal.push_back(s);
    }
    std::vector<double>  vectExponentiel;
    std::vector<double>  vectPredictions;

    double  sumExp = 0.0;
    double t = 0.0;

    for (int i = 0; i < ND; i++)
    {
        t = exp( mTotal[i]);
         vectExponentiel.push_back(t);
         sumExp += (t);
    }

    for (int i = 0; i < ND; i++)
    {
        t = ((double) vectExponentiel[i] / (double) sumExp);
         vectPredictions.push_back(t);
    }

    cache_making(); //Créer le cache de la dernière entrée

    return predictions;
}

std::vector<std::vector<double>> Softmax::backProp(const std::vector<double>&  dloss_dlayer_output, const double  learnRate)
{
    // dloss_dlayer_output est le gradient de perte pour les sorties de cette couche
    std::vector<std::vector<double>> d_L_d_inputs_shaped;

    for (int i = 0; i <  ND; i++)
    {
        //Seul 1 élément de dloss_dlayer_output sera différent de zéro
        if ( dloss_dlayer_output[i] == 0)
            continue;

       //Compter les exp^totals et Somme de tous les exp^totals
        std::vector<double>  totalExp;
        double s = 0.0;
        double t = 0.0;
        for (size_t i = 0; i < ND; i++) {
            t = exp( mCachedTotal[i]);
             totalExp.push_back(t);
            s += (t);
        }

        //Gradients de out[i] par rapport aux total
        std::vector<double> d_out_d_t;
        for (size_t j = 0; j <  ND; j++) {
            d_out_d_t.push_back((- totalExp[i]) *  totalExp[j] / (double)(pow(s, 2)));
        }

        d_out_d_t[i] =  totalExp[i] * (s -  totalExp[i]) / (double)(pow(s, 2));

        // Gradients des total par rapport aux poids/biais/entrée
        std::vector<double> d_t_d_w =  mCachedFlatten;

        double d_t_d_b = 1.0;

        std::vector<std::vector<double>> d_t_d_inputs =  mWeights;

        // Gradients de perte par rapport aux total
        std::vector<double>  dloss_dlayer_total;
        for (size_t j = 0; j <  ND; j++) {
             dloss_dlayer_total.push_back( dloss_dlayer_output[i] * d_out_d_t[j]);
        }

        //Gradients de perte par rapport aux poids/biais/entrée
        std::vector<std::vector<double>>  dloss_dlayer_weights;
        for (int k = 0; k <  mCachedLength; k++) {
            std::vector<double> sum;
            for (int j = 0; j <  ND; j++) {
                double s = 0;
                s += (d_t_d_w[k] *  dloss_dlayer_total[j]);
                sum.push_back(s);
            }
             dloss_dlayer_weights.push_back(sum);
        }

        std::vector<double>  dloss_dlayer_biases;
        for (size_t j = 0; j <  ND; j++) {
             dloss_dlayer_biases.push_back( dloss_dlayer_total[j] * d_t_d_b);
        }

        std::vector<double> d_L_d_inputs;
        for (int i = 0; i <  mCachedLength; i++) {
            double s = 0.0;
            for (int j = 0; j <  ND; j++) {
                s += (d_t_d_inputs[i][j] *  dloss_dlayer_total[j]);
            }
            d_L_d_inputs.push_back(s);
        }

        //Update weights / biases
        for (int k = 0; k <  mWeights.size(); k++)
        {
            for (int j = 0; j <  mWeights[k].size(); j++)
            {
                 mWeights[k][j] -= ( learnRate *  dloss_dlayer_weights[k][j]);
            }
        }
        for (int k = 0; k <  mBiases.size(); k++)
        {
             mBiases[k] -= ( learnRate *  dloss_dlayer_biases[k]);
        }

        //Nous devons reshape() avant de retourner d_L_d_inputs car nous avons aplati l'entrée lors de la passe avant
        for (int k = 0; k < DEPTH; k++)
        {
            std::vector<double> input;
            for (size_t p = 0; p < ( mCachedLength / DEPTH); p++) {
                input.push_back(d_L_d_inputs[k * ( mCachedLength / DEPTH) + p]);
            }
            d_L_d_inputs_shaped.push_back(input);
        }

        return d_L_d_inputs_shaped;

    }
    return d_L_d_inputs_shaped;
}    



