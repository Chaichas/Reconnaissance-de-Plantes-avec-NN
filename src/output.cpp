#include <iostream>
#include <string>
#include <dirent.h>
#include <algorithm>
#include <memory>
#include <ctime>
//For back_inserter
#include <iterator>

#include <mpi.h>

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
    //AM : Getting the MPI rank and size 
    int rank, comm_size;
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Comm_size(MPI_COMM_WORLD,&comm_size);

    m_convol->convolution_parameters(m_image->get_fusion_canal(), hauteur, largeur);
    m_pool->Pooling_parameters(m_convol->getConvMat(), m_convol->getMatHeight(), m_convol->getMatWidth());

// --------------------- AM: End of parallel Calculation --------------- 

	// AM: Gathering the output of the max_pooling

    // AM: Total size of the pooling image and the convolution image
    int pool_size = m_pool->getPoolingHeight() * m_pool->getPoolingWidth();
	int convol_size = m_convol->getMatHeight() * m_convol->getMatWidth();

    //fprintf(stderr, "pool_size %d, convol_size %d, pHeight %d hauteur %d \n", pool_size, convol_size, m_pool->getPoolingHeight(),hauteur);

    //AM: Computing counts and displ arrays to be used later for Gatherv
    int counts[comm_size];
    int displ[comm_size];
    displ[0] = 0;
    int global_size = 0;
    int local_height;
    m_pool->Pooling_height = 0;
    for (int i_rank = 0; i_rank < comm_size; i_rank++) {
        local_height = (m_image->local_images_height[i_rank]-2)/2;
        counts[i_rank] = local_height*m_pool->getPoolingWidth();
        m_pool->Pooling_height += local_height;
        global_size += counts[i_rank];
        if (i_rank > 0) displ[i_rank] = displ[i_rank-1]+counts[i_rank-1];
    }
    std::vector<double> filter_proc_mat(pool_size);
    std::vector<double> global_vec (global_size);
    std::vector<std::vector<double>> new_global_Matrix (filter_number, std::vector<double> (global_size));
    for (size_t ii = 0; ii < filter_number; ii++) {
        //std::copy(&m_pool->Pooling_Matrix[ii][0], &m_pool->Pooling_Matrix[ii][pool_size], filter_proc_mat.begin());
		MPI_Allgatherv(&m_pool->Pooling_Matrix[ii][0], pool_size, MPI_DOUBLE, &new_global_Matrix[ii][0], counts, displ, MPI_DOUBLE, MPI_COMM_WORLD);
        //new_global_Matrix[] = global_vec;
    }
    //fprintf(stderr,"first Allgatherv okay global_size %d\n",global_size);
    MPI_Barrier(MPI_COMM_WORLD);
    m_pool->Pooling_Matrix.clear();
    m_pool->Pooling_Matrix.resize(filter_number, std::vector<double> (global_size));
    m_pool->Pooling_Matrix.assign(new_global_Matrix.begin(), new_global_Matrix.end());
    /*for (size_t ii = 0; ii < filter_number; ii++) {
		//m_pool->Pooling_Matrix[ii].resize(global_size);
        std::copy(&new_global_Matrix[ii][0], &new_global_Matrix[ii][global_size], m_pool->Pooling_Matrix[ii].begin());
    }*/
	
	// AM: Assembly of the hidden matrix of convolution layer
    m_convol->ConvMat_height = 0;
    global_size = 0;
    for (int i_rank = 0; i_rank < comm_size-1; i_rank++) {
        local_height = m_image->local_images_height[i_rank]-2;
        counts[i_rank] = local_height*largeur;
        global_size += counts[i_rank];
        m_convol->ConvMat_height += local_height;
        if (i_rank > 0) displ[i_rank] = displ[i_rank-1]+counts[i_rank-1];
    }
    m_convol->ConvMat_height += m_image->local_images_height[comm_size-1] - 2; 
	counts[comm_size-1] = m_image->local_images_height[comm_size-1]*largeur;
    global_size += counts[comm_size-1];
    
    if(comm_size > 1) displ[comm_size-1] = displ[comm_size-2]+counts[comm_size-2];
    //fprintf(stderr,"global_size %d m_convol->HiddenMat.size() %d counts[0] %d displ[0] %d\n",global_size, m_convol->HiddenMat.size(),counts[0],displ[0]);
    global_vec.clear();
	global_vec.resize(global_size);
	MPI_Allgatherv(&m_convol->HiddenMat[0], m_convol->HiddenMat.size(), MPI_DOUBLE, &global_vec[0], counts, displ, MPI_DOUBLE, MPI_COMM_WORLD);
    //fprintf(stderr,"second Allgatherv okay \n");
	MPI_Barrier(MPI_COMM_WORLD);
    m_convol->HiddenMat.clear();
    m_convol->HiddenMat.resize(global_size);
	m_convol->HiddenMat.assign(global_vec.begin(), global_vec.end());
    
	// AM: Assembly of the hidden matrix of pooling layer
    global_size = 0;
	for (int i_rank = 0; i_rank < comm_size; i_rank++) {
        counts[i_rank] = (m_image->local_images_height[i_rank]-2)*m_convol->getMatWidth();
        global_size += counts[i_rank];
        if (i_rank > 0) displ[i_rank] = displ[i_rank-1]+counts[i_rank-1];
    }
    filter_proc_mat.resize(convol_size);
    global_vec.clear();
    global_vec.resize(global_size);
    new_global_Matrix.clear();
    new_global_Matrix.resize(filter_number, std::vector<double> (global_size));
    for (size_t ii = 0; ii < filter_number; ii++) {
        //std::copy(&m_pool->HiddenMat_input[ii][0], &m_pool->HiddenMat_input[ii][convol_size], filter_proc_mat.begin());
		MPI_Allgatherv(&m_pool->HiddenMat_input[ii][0], convol_size, MPI_DOUBLE, &new_global_Matrix[ii][0], counts, displ, MPI_DOUBLE, MPI_COMM_WORLD);
		//std::copy(&global_vec[0], &global_vec[global_size], new_global_Matrix[ii].begin());
    }
    //fprintf(stderr,"third Allgatherv okay \n");
    MPI_Barrier(MPI_COMM_WORLD);
    m_pool->HiddenMat_input.clear();
    m_pool->HiddenMat_input.resize(filter_number, std::vector<double> (global_size));
    m_pool->HiddenMat_input.assign(new_global_Matrix.begin(), new_global_Matrix.end());
    /*
    for (size_t ii = 0; ii < filter_number; ii++) {
        std::copy(&new_global_Matrix[ii][0], &new_global_Matrix[ii][global_size], m_pool->HiddenMat_input[ii].begin());
    }
    */
    filter_proc_mat.clear();
    filter_proc_mat.shrink_to_fit();

    global_vec.clear();
    global_vec.shrink_to_fit();

    new_global_Matrix.clear();
    new_global_Matrix.shrink_to_fit();


    //--------------- AM: Black Box starts here (like in the sequential calculation) ------------------------------------------------------------------
    //fprintf(stderr,"starting softmax \n");
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

//-------------------------------------------Training Part------------------------------------------------------

void output::Training_data(int numb_epoch, double alpha)
{
    //AM: Getting the MPI rank and size
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Comm_size(MPI_COMM_WORLD,&size);

    //AM: Display on only one proc
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

            //Initialisation de gradient
            std::vector<double> gradient = { 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0 };
            gradient[Labels[label_i]] = (-1.0 / proba[Labels[label_i]]);

            std::vector<std::vector<double>> gradient_result = m_softmax->BackPropagation(gradient, alpha);
            std::vector<std::vector<double>> pool_result = m_pool->BackPropagation(gradient_result);
            m_convol->BackPropagation(pool_result, alpha);

            runningLoss += loss;
            runningAcc += acc;
        }

        label_i++;

        //Affichage de perte et de la precison pour chaque epoch 
        //AM: Display on only one proc
        if(rank == 0)
            std::cout << "Epoch " << it << " : Average Loss " << runningLoss / label_i << " , Accuracy " << (runningAcc / label_i) * 100 << " %" << '\n';
        runningLoss = 0.0;
        runningAcc = 0.0;
        it++;
    }
}

//-------------------------------------------Testing Part------------------------------------------------------

void output::Testing_data()
{
    //AM: Getting rank and size
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Comm_size(MPI_COMM_WORLD,&size);

    int label_i = 0;
    double runningAcc = 0.0, runningLoss = 0.0;

    //AM: Displaying on only one proc
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


        runningAcc += acc;
        runningLoss += loss;

        int predIndex = std::distance(out.begin(), std::max_element(out.begin(), out.end()));

        if (labels_test[label_i] == predIndex)
            right++;

    }
    label_i++;

    int wrong = label_i - right;
    if(rank == 0) {
        //AM: Displaying on only one proc
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


