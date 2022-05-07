﻿#include "../include/softmax_layer.h"
#include <algorithm>
#include "../include/Convolution_layer.h"

void Softmax_layer::Flatten(const std::vector<std::vector<double>>& img_input, int d)
{
    int idx = 0;
    for (int i = 0; i < d; i++) {
        for (int jj = 0; jj < img_input[i].size(); jj++) {
            mFlatten[idx] = img_input[i][jj];
            idx++;
        }
    }

}

void Softmax_layer::Hidden()
{
    //Effacer le dernier cache et redimensionner
    //mCachedFlatten.clear();
    //mCachedTotal.clear();
    //mCachedFlatten.resize(mFlatten.size());
    //mCachedTotal.resize(mTotal.size());

    //Copier la derniére entrée et les paramétres
    mCachedLength = mLength;
    mCachedFlatten.assign(mFlatten.begin(), mFlatten.end());
    mCachedTotal.assign(mTotal.begin(), mTotal.end());

}



std::vector<double> Softmax_layer::Softmax_start(const std::vector<std::vector<double>>& img_input, int img_height, int img_width)
{
    mLength = img_height * img_width * DEPTH;
    

    //Pour l'initialisation des poids et des biais en premiére exécution
    if (pred)
    {
        //Attribuez 0 à mBiases (la longueur de m_biase est de 5)
        mBiases.assign(ND, 0.0);

        //Distribution de nombres aléatoires qui produit des valeurs  virgule flottante selon une distribution normale
        mWeights.resize(mLength, std::vector<double> (ND)); // AM: initialisation de la taille du filtre suite à la modification de random_weights
        Convolution_layer::random_weights(mLength, ND, mWeights);


        for (size_t i = 0; i < mLength; i++) {
            for (size_t j = 0; j < ND; j++) {
                mWeights[i][j] = (double)mWeights[i][j] / (double)mLength;
            }
        }

        pred = false;
        mFlatten.resize(DEPTH*img_height*img_width);
        mCachedFlatten.resize(mFlatten.size());
        mTotal.resize(ND);
        mCachedTotal.resize(mTotal.size());
    }

    //mFlatten.clear();//Effacer le dernier aplatissement de l'entrée
    //mTotal.clear();//Effacer la derniere prédiction 


    Flatten(img_input, DEPTH);

    //Multiplier l'entrée aplatie et mWeights
    int i = 0;
    while (i < ND){
        double s = 0;
        // Boucle pour multiplier  mWeights[j] pour chaque  plante avec chaque mFlatten [j]
        for (int j = 0; j < mLength; j++) {
            s = s + (mFlatten[j] * mWeights[j][i]);
        }
        //Somme de biais
        s = s + mBiases[i];
        //mTotal.push_back(s);
        mTotal[i] = s;
        i++;
    }
    std::vector<double>  vectExponentiel (ND);
    std::vector<double>  vectPredictions (ND);

    double  sumExp = 0.0;
    double t = 0.0;
    i = 0;
    while( i < ND)
    {
        t = exp(mTotal[i]);
        //vectExponentiel.push_back(t);
        vectExponentiel[i] = t;
        sumExp = sumExp + t;
        i++;
    }
    int j = 0;
    while (j < ND)
    {
        t = ((double)vectExponentiel[j] / (double)sumExp);
        //vectPredictions.push_back(t);
        vectPredictions[j] = t;
        j++;
    }

    Hidden(); //Créer le cache de la derniére entrée

    return vectPredictions;
}




std::vector<std::vector<double>> Softmax_layer::BackPropagation(const std::vector<double>& dLossdOut, const double learn_rate)
{
    //dLossdOut is the loss gradient for this layer's outputs
    std::vector<std::vector<double>> dlossdinputs_shaped;

    for (int i = 0; i < ND; i++)
    
    {
        //Seul 1 élément de dloss_dlayer_output sera différent de zéro
        if (dLossdOut[i] == 0)
            continue;

        //Compter les exp^totals et Somme de tous les exp^totals
        std::vector<double> t_exp (ND);
        double sum = 0.0;
        double temp = 0.0;
        for (int i = 0; i < ND; i++) {
            temp = exp(mCachedTotal[i]);
            //t_exp.push_back(temp);
            t_exp[i] = temp;
            sum += (temp);
        }

        // Gradients des total par rapport aux total
        std::vector<double> dOutdt(ND);
        for (int j = 0; j < ND; j++) {
            //dOutdt.push_back((-t_exp[i]) * t_exp[j] / (double)(pow(sum, 2)));
            dOutdt[j] = (-t_exp[i]) * t_exp[j] / (double)(pow(sum, 2));
        }

        dOutdt[i] = t_exp[i] * (sum - t_exp[i]) / (double)(pow(sum, 2));

        // Gradients des total par rapport aux poids/biais/entrée
        std::vector<double> dtdw = mCachedFlatten;

        double dtdb = 1.0;

        std::vector<std::vector<double>> dtdinputs = mWeights;

        // Gradients de perte par rapport aux total
        std::vector<double> dLossdt(ND);
        for (size_t j = 0; j < ND; j++) {
            //dLossdt.push_back(dLossdOut[i] * dOutdt[j]);
            dLossdt[j] = dLossdOut[i] * dOutdt[j];
        }

        std::vector<double> dlossdinputs (mCachedLength);
        for (int i = 0; i < mCachedLength; i++) {
            double sum = 0.0;
            for (int j = 0; j < ND; j++) {
                sum = sum + (dtdinputs[i][j] * dLossdt[j]);
            }
            //dlossdinputs.push_back(sum);
            dlossdinputs[i] = sum;
        }
        // Gradients de perte par rapport aux biais
        std::vector<double> dLossdb (ND);
        for (int j = 0; j < ND; j++) {
            //dLossdb.push_back(dLossdt[j] * dtdb);
            dLossdb[j] = dLossdt[j] * dtdb;
        }

        // Gradients de perte par rapport aux poids
        std::vector<std::vector<double>> dLossdw (mCachedLength, std::vector<double> (ND));
        for (int k = 0; k < mCachedLength; k++) {
            for (int j = 0; j < ND; j++) {
                double sum = 0;
                sum = sum + (dtdw[k] * dLossdt[j]);
                //vectsum.push_back(sum);
                dLossdw[k][j] = sum;
            }
            //dLossdw.push_back(vectsum);

        }

       

        //mettre à jour les poids et les biais
        for (int k = 0; k < mWeights.size(); k++)
        {
            for (int j = 0; j < mWeights[k].size(); j++)
            {
                mWeights[k][j] = mWeights[k][j] - (learn_rate * dLossdw[k][j]);
            }
        }
        for (int k = 0; k < mBiases.size(); k++)
        {
            mBiases[k] = mBiases[k] -(learn_rate * dLossdb[k]);
        }

        //Nous devons reshape() avant de retourner d_L_d_inputs car nous avons aplati l'entrée lors de la passe avant
        for (int k = 0; k < DEPTH; k++)
        {
            std::vector<double> input;
            for (size_t p = 0; p < (mCachedLength / DEPTH); p++) {
                input.push_back(dlossdinputs[k * (mCachedLength / DEPTH) + p]);
            }
            dlossdinputs_shaped.push_back(input);
        }

        
        return dlossdinputs_shaped;
    }
    
    return dlossdinputs_shaped;
}

