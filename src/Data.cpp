#include <iostream>
#include <string>
#include "../include/Data.h"
#include <mpi.h>

Data::Data() {}
Data::~Data() {}

void Data::decomposition(const int comm_size, const int total_rows, std::vector<int>& local_images_height)
{
    int rows_output; // hauteur de la matrice à la sortie de max_pooling;
    rows_output = (total_rows-2)/2; // hypothèses: total_rows est paire

    // La décomposition est basé sur rows_output
	for (int i_rank=0; i_rank<comm_size; i_rank++) {
        // décomposition optimale pour load balance
		if (i_rank < rows_output%comm_size) {
			local_images_height[i_rank] = (rows_output/comm_size)+1;
		}
		else {
			local_images_height[i_rank] = rows_output/comm_size;
		}
       
        // chaque ligne de la matrice max_pooling correspond à 2 ligne de la matrice convolution
        local_images_height[i_rank] *= 2;
        
        // le fitre de convolution a besoin de deux lignes supplémentaires
        local_images_height[i_rank] += 2;
        //fprintf(stderr, "local_image_height %d \n",local_images_height[i_rank]);
	}
}

void Data::loadImage(const std::string& str_Path, int& hauteur, int& largeur)
{
    //Recuperation de l'image
    int rank, comm_size;
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Comm_size(MPI_COMM_WORLD,&comm_size);

    Mat* image = new Mat();

    local_images_height.resize(comm_size);
    //int local_images_height[comm_size]; //Tab contenant la taille des images locales
    int displ[comm_size]; //position de l'image locale ds celle globale
    int channels;
    int elmt_size;
    int cols;

    if (rank == 0){

        m_ImageVector.clear();
       
        //Creation et lecture de l'image par un seul proc
        (*image) = imread(str_Path, IMREAD_COLOR); //imread from opencv
        channels = image->channels();
        elmt_size = image->step[0];
        cols = image->cols;
        decomposition(comm_size, image->rows, local_images_height); //cette fonction retourne la hauteur de chaque image locale et sa position dans l'image complete
        //fprintf(stderr,"local_height before Bcast %d \n",local_images_height[0]);
    }
    MPI_Bcast(&local_images_height[0],comm_size,MPI_INT,0,MPI_COMM_WORLD);
    MPI_Bcast(&channels, 1, MPI_INT, 0, MPI_COMM_WORLD );
    MPI_Bcast(&elmt_size, 1, MPI_INT, 0, MPI_COMM_WORLD );
    MPI_Bcast(&cols, 1, MPI_INT, 0, MPI_COMM_WORLD );
    MPI_Barrier( MPI_COMM_WORLD ); //synchronisation de procs
    //fprintf(stderr,"local_height after Bcast %d \n",local_images_height[0]);
    int local_images_sizes[comm_size];
    displ[0] = 0;
    for (int i_rank = 0; i_rank < comm_size; i_rank++) {
        local_images_sizes[i_rank] = local_images_height[i_rank]*elmt_size;
        if (i_rank > 0) displ[i_rank] = displ[i_rank-1]+local_images_sizes[i_rank-1]-2*elmt_size;
    }
    uchar* local_buffer;
    local_buffer = new uchar[local_images_sizes[rank]];
    MPI_Barrier( MPI_COMM_WORLD ); //synchronisation de procs, pour avoir sure que les processeurs ont alloce le buffer

    // scatterv les images entre les procs (taille d'images locales variables)
    MPI_Scatterv( image->data, local_images_sizes, displ, MPI_UNSIGNED_CHAR, local_buffer, local_images_sizes[rank], MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD );

    hauteur = local_images_height[rank];
    //fprintf(stderr,"hauteur %d \n",hauteur);
    largeur = cols;
    // synchronisation de traitement d image
    //MPI_Barrier( MPI_COMM_WORLD );
    
    /*
    if(!image.data)
    {
          std::cout << "Error: the image wasn't correctly loaded." << std::endl;
          return -1;
    }
    */
    //Recuperation des dimension de l'image
    //hauteur = image->rows;
    //largeur = image->cols;

    //Generation de vecteur RVB
    //this->create_canal(image);
    double R,V,B,somme;
    for ( size_t i = 0; i < local_images_sizes[rank]; i += channels ) {
        R = (double)local_buffer[i];
        V = (double)local_buffer[i+1];
        B = (double)local_buffer[i+2];
        somme = (R + V + B);
        //normalisation des valeurs de pixels dans la plage [-0.5, 0.5] pour ne pas ralentir le processus d'apprentissage
        somme = (somme *(1/ 765)) - 0.5;
        m_ImageVector.push_back(somme);
    }

    delete image;
}

/*
void Data::create_canal(Mat* image)
{
    int i = 0;
    while (i < image->rows)
    {
        int j = 0;
	double R,V,B,somme;
        while (j < image->cols)
        {
	        R = ((double)image->ptr<Vec3b>(i)[j][2]);
            V = ((double)image->ptr<Vec3b>(i)[j][1]);
            B = ((double)image->ptr<Vec3b>(i)[j][0]);
            somme = (R + V + B);

            //normalisation des valeurs de pixels dans la plage [-0.5, 0.5] pour ne pas ralentir le processus d'apprentissage
            somme = (somme *(1/ 765)) - 0.5;
            m_ImageVector.push_back(somme);
            j++;
        }
        i++;
    }
} */

const std::vector<double>& Data::get_fusion_canal() const
{
    return m_ImageVector;
}

/*
     Reference : - https://machinelearningmastery.com/how-to-manually-scale-image-pixel-data-for-deep-learning/
                 - https://sodocumentation.net/opencv/topic/1957/pixel-access
*/
