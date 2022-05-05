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
using namespace std;


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


    m_convol->convolution_parameters(m_image->get_fusion_canal(), hauteur, largeur);
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

//-------------------------------------------Training Part------------------------------------------------------

void output::transform_matrix_to_vector(const std::vector<std::vector<double>> &matrix, std::vector<double>& simplified_vector, const int dim1, const int dim2) {
    size_t idx = 0;
    for (int ii = 0; ii < dim1; ii++) {
        for (int jj = 0; jj < dim2; jj++) {
            simplified_vector[idx] = matrix[ii][jj];
            idx++;
        }
    }
}

void output::transform_matrix3_to_vector(const std::vector<std::vector<std::vector<double>>> &matrix, std::vector<double>& simplified_vector, const int dim1, const int dim2, const int dim3) {
    size_t idx = 0;
    for (int ii = 0; ii < dim1; ii++) {
        for (int jj = 0; jj < dim2; jj++) {
			for (int kk = 0; kk < dim3; kk++) {
				simplified_vector[idx] = matrix[ii][jj][kk];
				idx++;
			}
        }
    }
}

void output::transform_vector_to_matrix(std::vector<std::vector<double>>& matrix, const std::vector<double> &simplified_vector, const int dim1, const int dim2) {
    for (int ii = 0; ii < dim1; ii++) {
		for (int jj = 0; jj < dim2; jj++) {
			matrix[ii][jj] = simplified_vector[ii*dim2+jj];
		}
	}
}

void output::transform_vector_to_matrix3(std::vector<std::vector<std::vector<double>>>& matrix, const std::vector<double> &simplified_vector, const int dim1, const int dim2, const int dim3) {
    for (int ii = 0; ii < dim1; ii++) {
		for (int jj = 0; jj < dim2; jj++) {
			for (int kk = 0; kk < dim3; kk++) {
				matrix[ii][jj][kk] = simplified_vector[(ii*dim2+jj)*dim3+kk];
			}
		}
	}
}

void output::Training_data(int numb_epoch, double alpha)
{
	
    int rank, comm_size;
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Comm_size(MPI_COMM_WORLD,&comm_size);
	
    if (rank == 0)
        std::cout << "--------------Start Training ----------" << '\n';

    std::vector<int> Labels;
    std::vector<std::string> training_files = output::Process_directory(m_trainPath, Labels);
	int tot_files = training_files.size();
	

    int it_ep = 0;
    std::vector<vector<double>> proc_images;
    std::vector<vector<vector<double>>> proc_conv_images;
    std::vector<vector<vector<double>>> proc_pool_images;

    std::vector<std::vector<double>> global_images;
    std::vector<std::vector<std::vector<double>>> global_conv_images;
    std::vector<std::vector<std::vector<double>>> global_pool_images;

    std::vector<double> simplified_vector;
    std::vector<double> global_vector;

    int fs_rank, fe_rank;
    if (rank < tot_files%comm_size)
    {
        fs_rank = (tot_files/comm_size)*rank + rank;
        fe_rank = (tot_files/comm_size)*(rank+1) + rank + 1;
    }
    else
    {
        fs_rank = (tot_files/comm_size)*rank + tot_files%comm_size;
        fe_rank = (tot_files/comm_size)*(rank+1) + tot_files%comm_size;
    }
    int file_counts[comm_size];
    int file_displ[comm_size];
    file_displ[0] = 0;
    for (int i_rank=0; i_rank<comm_size; i_rank++) {
        if (i_rank < tot_files%comm_size) {
            file_counts[i_rank] = (tot_files/comm_size)+1;
        }
        else {
            file_counts[i_rank] = tot_files/comm_size;
        }
        if (i_rank > 0) file_displ[i_rank] = file_displ[i_rank-1]+file_counts[i_rank-1];
    }
    int file_size = fe_rank - fs_rank;

    int count[comm_size];
    int displ[comm_size];

    int image_size;

    while (it_ep < numb_epoch) {

        int label_i = 0;
        double runningAcc = 0.0, runningLoss = 0.0;

        int conv_h, pool_h;
		
        int hauteur, largeur;
        if (it_ep == 0) {
        clock_t begin_time = std::clock();
        for (int file_idx = fs_rank; file_idx < fe_rank; file_idx++)
        {
            std::string name = training_files[file_idx];
            hauteur = 0; 
			largeur = 0;

            m_image->loadImage(name, hauteur, largeur);
            proc_images.push_back(m_image->m_ImageVector);
            image_size = hauteur*largeur;
            

            //Lancement de training
			
			m_convol->convolution_parameters(m_image->get_fusion_canal(), hauteur, largeur);
			conv_h = m_convol->getMatHeight();
            //fprintf(stderr,"H %d W %d \n",m_convol->getMatHeight(),m_convol->getMatWidth());
            m_pool->Pooling_parameters(m_convol->getConvMat(), m_convol->getMatHeight(), m_convol->getMatWidth());
			pool_h = m_convol->getMatWidth();
			proc_conv_images.push_back(m_convol->getConvMat());
			proc_pool_images.push_back(m_pool->getPoolingMatrix());
		}

		int conv_size = proc_conv_images[0][0].size();
		int pool_size = proc_pool_images[0][0].size();

        simplified_vector.resize(file_size*image_size);
        global_vector.resize(tot_files*image_size);
        transform_matrix_to_vector(proc_images, simplified_vector, file_size, image_size);
        for (int i_rank =0; i_rank < comm_size; i_rank++) {
            count[i_rank] = file_counts[i_rank]*image_size;
            displ[i_rank] = file_displ[i_rank]*image_size;
        }
        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Gatherv(&simplified_vector[0], simplified_vector.size(), MPI_DOUBLE, &global_vector[0], count, displ, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        if (rank == 0) {
            global_images.resize(tot_files, std::vector<double> (image_size));
            transform_vector_to_matrix(global_images, global_vector, tot_files, image_size);
        }
		
		for (int i_rank =0; i_rank < comm_size; i_rank++) {
			count[i_rank] = file_counts[i_rank]*conv_size*8;
			displ[i_rank] = file_displ[i_rank]*conv_size*8;
		}
        simplified_vector.clear();
        global_vector.clear();
		simplified_vector.resize(file_size*conv_size*8); //8 nbr of filters
		global_vector.resize(tot_files*conv_size*8); //8 nbr of filters
		transform_matrix3_to_vector(proc_conv_images, simplified_vector,file_size,8,conv_size);

        MPI_Barrier(MPI_COMM_WORLD);
		MPI_Gatherv(&simplified_vector[0], simplified_vector.size(), MPI_DOUBLE, &global_vector[0], count, displ, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        if (rank == 0) {
			global_conv_images.resize(tot_files, std::vector<std::vector<double>> (8, std::vector<double> (conv_size)));
			transform_vector_to_matrix3(global_conv_images, global_vector, tot_files, 8, conv_size);
		}
		
		for (int i_rank =0; i_rank < comm_size; i_rank++) {
			count[i_rank] = file_counts[i_rank]*pool_size*8;
			displ[i_rank] = file_displ[i_rank]*pool_size*8;
		}
        simplified_vector.clear();
        global_vector.clear();
		simplified_vector.resize(file_size*pool_size*8); //8 nbr of filters
		global_vector.resize(tot_files*pool_size*8); //8 nbr of filters
		transform_matrix3_to_vector(proc_pool_images, simplified_vector,file_size,8,pool_size);
        MPI_Barrier(MPI_COMM_WORLD);
		MPI_Gatherv(&simplified_vector[0], simplified_vector.size(), MPI_DOUBLE, &global_vector[0], count, displ, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        if (rank == 0) {
			global_pool_images.resize(tot_files, std::vector<std::vector<double>> (8, std::vector<double> (pool_size)));
			transform_vector_to_matrix3(global_pool_images, global_vector, tot_files, 8, pool_size);
            fprintf(stderr,"pre-processing step took %f s \n",float( clock () - begin_time ) /  CLOCKS_PER_SEC);
        }

        }
		
		if (rank == 0) {
        
        for (std::vector<std::string>::iterator it = training_files.begin(); it != training_files.end(); ++it)
        {
		
			label_i = std::distance(training_files.begin(), it);
			
			m_convol->Hidden(global_images[label_i]);
			m_pool->Hidden(global_conv_images[label_i]);
			//fprintf(stderr,"global_images[30][06] %f,  HM %f \n",global_images[label_i][06],m_convol->HiddenMat[06]);
			std::vector<double> proba = m_softmax->Softmax_start(global_pool_images[label_i], m_pool->getPoolingHeight(), m_pool->getPoolingWidth());

			loss = -log(proba[Labels[label_i]]);

			//auto  max = std::max_element(proba.begin(), proba.end());
			int proba_i = std::distance(proba.begin(), std::max_element(proba.begin(), proba.end()));

			if (proba_i == Labels[label_i])
				acc = 1;
			else
				acc = 0;
			
            //std::vector<double> proba = prediction(Labels[label_i], hauteur, largeur);

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
        std::cout << "Epoch " << it_ep << " : Average Loss " << runningLoss / label_i << " , Accuracy " << (runningAcc / label_i) * 100 << " %" << '\n';
        runningLoss = 0.0;
        runningAcc = 0.0;
		}
        it_ep++;
    }
}

//-------------------------------------------Testing Part------------------------------------------------------

void output::Testing_data()
{

    int label_i = 0;
    double runningAcc = 0.0, runningLoss = 0.0;

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
    std::cout << "--------------------------------Result of testing-------------------------" << '\n';

    std::cout << "Average Loss" << runningLoss / label_i << " , Accuracy " << '\n';
    std::cout << "Accuracy " << (runningAcc / label_i) * 100 << " %." << '\n';

    std::cout << "---------------------------------------------------------------------------" << '\n';
    std::cout << "Le nombre d'image de test est : " << label_i << '\n';
    std::cout << "Le nombre d'image correctement predits : " << right << '\n';
    std::cout << "Le nombre d'image non predits : " << wrong << '\n';

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


