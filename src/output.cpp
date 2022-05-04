#include <iostream>
#include <string>
#include <dirent.h>
#include <algorithm>
#include <memory>
#include <ctime>
//For back_inserter
#include <iterator>

#include "../include/Output.h"



output::output(std::string file_train, std::string file_test) : m_trainPath(file_train), m_testPath(file_test)
{
    m_image = new Data();
    m_convol = new Convolution_layer();
    m_pool = new Pooling_layer();
    m_softmax = new Softmax_layer();
}


output::~output()
{
    delete m_image;
    delete m_pool;
    delete m_softmax;
}
//-------------------------Train-------------------------------

/*void output::train(int c, int& hauteur, int& largeur, double& lRate)
{
    std::vector<double> proba = prediction(c, hauteur, largeur);

    //Initialisation de gradient
    std::vector<double> gradient = { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
    gradient[c] = (-1.0 / proba[c]);

    std::vector<std::vector<double>> gradient_result = m_softmax->BackPropagation(gradient, lRate);
    std::vector<std::vector<double>> pool_result = m_pool->BackPropagation(gradient_result);
    m_convol->BackPropagation(pool_result, lRate);
}
*/
//-------------------------------------------Prediction Part------------------------------------------------------

std::vector<double> output::prediction(int c, int& hauteur, int& largeur)
{
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);

    m_convol->convolution_parameters(m_image->get_fusion_canal(), hauteur, largeur);
    if (rank==0) {
    m_pool->Pooling_parameters(m_convol->getConvMat(), m_convol->getMatHeight(), m_convol->getMatWidth());
    std::vector<double> proba = m_softmax->Softmax_start(m_pool->getPoolingMatrix(), m_pool->getPoolingHeight(), m_pool->getPoolingWidth());

    loss = -log(proba[c]);

    //auto  max = std::max_element(proba.begin(), proba.end());
    int proba_i = std::distance(proba.begin(), std::max_element(proba.begin(), proba.end()));

    if (proba_i == c)
        acc = 1;
    else
        acc = 0;

    return proba;
    }
    else{
        std::vector<double> v_null;
        return v_null;
    }
}

//-------------------------------------------Training Part------------------------------------------------------

void output::Training_data(int numb_epoch, double alpha)
{
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);

    if(rank==0) 
        std::cout << "--------------Start Training ----------" << '\n';

    std::vector<int> Labels;
    std::vector<std::string> training_files = output::Process_directory(m_trainPath, Labels);

    int it = 0;
    while (it < numb_epoch) {

        int label_i = 0;
        double runningAcc = 0.0, runningLoss = 0.0;
        
        for (std::vector<std::string>::iterator it = training_files.begin(); it != training_files.end(); ++it)
        {
            std::string name = *it;
            int hauteur = 0, largeur = 0;

            m_image->loadImage(name, hauteur, largeur);
            label_i = std::distance(training_files.begin(), it);

            //Lancement de training
           
            std::vector<double> proba = prediction(Labels[label_i], hauteur, largeur);
            if (rank == 0) {
            //Initialisation de gradient
            std::vector<double> gradient = { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
            gradient[Labels[label_i]] = (-1.0 / proba[Labels[label_i]]);

            std::vector<std::vector<double>> gradient_result = m_softmax->BackPropagation(gradient, alpha);
            std::vector<std::vector<double>> pool_result = m_pool->BackPropagation(gradient_result);
            m_convol->BackPropagation(pool_result, alpha);

            runningLoss += loss;
            runningAcc += acc;
            }
        }
        label_i++;
        if (rank == 0) {
        

        //Affichage de perte et de la precison pour chaque epoch 
        if(rank == 0)
            std::cout << "Epoch " << it << " : Average Loss " << runningLoss / label_i << " , Accuracy " << (runningAcc / label_i) * 100 << " %" << '\n';
        runningLoss = 0.0;
        runningAcc = 0.0;
        }
        it++;
    }
}

//-------------------------------------------Testing Part------------------------------------------------------

void output::Testing_data()
{

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);

    int label_i = 0;
    double runningAcc = 0.0, runningLoss = 0.0;

    if(rank == 0)
        std::cout << "-----------------Testing Data -------------------" << '\n';

    std::vector<int> labels_test;
    std::vector<std::string> testing_files = output::Process_directory(m_trainPath, labels_test);

    int right = 0;
    for (std::vector<std::string>::iterator it = testing_files.begin(); it != testing_files.end(); ++it)
    {
        std::string name = *it;
        int hauteur = 0, largeur = 0;

        m_image->loadImage(name, hauteur, largeur);
        label_i = std::distance(testing_files.begin(), it);

        //Recuperation de vecteur de probabilte de sortie 
        std::vector<double> out = prediction(labels_test[label_i], hauteur, largeur);

        if (rank == 0) {
        runningAcc += acc;
        runningLoss += loss;

        int predIndex = std::distance(out.begin(), std::max_element(out.begin(), out.end()));

        if (labels_test[label_i] == predIndex)
            right++;
        }
    }
    label_i++;

    
    if(rank == 0) {
        int wrong = label_i - right;
        std::cout << "--------------------------------Result of testing-------------------------" << '\n';

        std::cout << "Average Loss" << runningLoss / label_i << " , Accuracy " << '\n';
        std::cout << "Accuracy " << (runningAcc / label_i) * 100 << " %." << '\n';

        std::cout << "---------------------------------------------------------------------------" << '\n';
        std::cout << "Le nombre d'image de test est : " << label_i << '\n';
        std::cout << "Le nombre d'image correctement predits : " << right << '\n';
        std::cout << "Le nombre d'image non predits : " << wrong << '\n';
    }

}


std::vector<std::string> output::Process_directory(const std::string& path, std::vector<int>& label) {

    DIR* dir;
    dirent* pDir;
    std::vector<std::string> temp_files;
    dir = opendir(path.c_str());
    while (pDir = readdir(dir)) {
        std::string name = pDir->d_name;
        int32_t pos = name.find(".");
        std::string extension = name.substr(pos+1);
        if (extension == "jpg") {
            std::string a = path + "/" + name;
            temp_files.push_back(a);
            //Get label
            char c = name[0];
            int ic = c - '0';
            label.push_back(ic);
        }
    }
    return temp_files;
}


