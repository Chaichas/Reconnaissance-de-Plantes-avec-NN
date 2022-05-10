#include <iostream>
#include <string>
#include "../include/Data.h"
#include <mpi.h>

Data::Data() {}
Data::~Data() {}

//AM: This function returns the length of every local image and its position in the complete image
void Data::decomposition(const int comm_size, const int total_rows, std::vector<int>& local_images_height)
{
    int rows_output; // AM: length of the matrix as output of the max_pooling;
    rows_output = (total_rows-2)/2; // AM: Assumption: total_rows is even

    // AM: Decomposition based on rows_output
	for (int i_rank=0; i_rank<comm_size; i_rank++) {
        // AM: Efficient decomposition for load balance
		if (i_rank < rows_output%comm_size) {
			local_images_height[i_rank] = (rows_output/comm_size)+1;
		}
		else {
			local_images_height[i_rank] = rows_output/comm_size;
		}
       
        // AM: Every line of the max_pooling matrix corresponds to 2 lines of the convolution matrix
        local_images_height[i_rank] *= 2;
        
        // AM: The convolution filter needs 2 additional lines
        local_images_height[i_rank] += 2;
        //fprintf(stderr, "local_image_height %d \n",local_images_height[i_rank]); //AM: Debugging
	}
}

void Data::loadImage(const std::string& str_Path, int& hauteur, int& largeur)
{
    //AM: This implementation is inspired from: https://gist.github.com/Xonxt/1ec2f58c2231d5c643dc83ddcd61e395
    //AM: Getting MPI rank and size
    int rank, comm_size;
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Comm_size(MPI_COMM_WORLD,&comm_size);

    Mat* image = new Mat();

    local_images_height.resize(comm_size);
    int displ[comm_size]; //AM: position of the local image in the global one
    int channels;
    int elmt_size;
    int cols;

    if (rank == 0){

        m_ImageVector.clear();
       
        //AM: Creation and reading of the image by one proc
        (*image) = imread(str_Path, IMREAD_COLOR); //imread from opencv
        channels = image->channels();
        elmt_size = image->step[0];
        cols = image->cols;
        //AM: This function returns the length of every local image and its position in the complete image
        decomposition(comm_size, image->rows, local_images_height);
        //fprintf(stderr,"local_height before Bcast %d \n",local_images_height[0]);
    }

    // AM: Broadcasting the needed information (image height width and channel size) for receiving the image data
    MPI_Bcast(&local_images_height[0],comm_size,MPI_INT,0,MPI_COMM_WORLD);
    MPI_Bcast(&channels, 1, MPI_INT, 0, MPI_COMM_WORLD );
    MPI_Bcast(&elmt_size, 1, MPI_INT, 0, MPI_COMM_WORLD );
    MPI_Bcast(&cols, 1, MPI_INT, 0, MPI_COMM_WORLD );
    MPI_Barrier( MPI_COMM_WORLD ); //AM: procs synchronization

    //fprintf(stderr,"local_height after Bcast %d \n",local_images_height[0]);
    //AM: Computing the count and displ arrays to be used by scatterv.
    int local_images_sizes[comm_size];
    displ[0] = 0;
    for (int i_rank = 0; i_rank < comm_size; i_rank++) {
        local_images_sizes[i_rank] = local_images_height[i_rank]*elmt_size;
        if (i_rank > 0) displ[i_rank] = displ[i_rank-1]+local_images_sizes[i_rank-1]-2*elmt_size;
    }
    uchar* local_buffer;
    local_buffer = new uchar[local_images_sizes[rank]];
    MPI_Barrier( MPI_COMM_WORLD ); //AM : Procs synchronization to be sure that procs allocated the buffer

    // AM: scatterv of images between procs (variable size of local images)
    MPI_Scatterv( image->data, local_images_sizes, displ, MPI_UNSIGNED_CHAR, local_buffer, local_images_sizes[rank], MPI_UNSIGNED_CHAR, 0, MPI_COMM_WORLD );

    hauteur = local_images_height[rank];
    //fprintf(stderr,"hauteur %d \n",hauteur);
    largeur = cols;

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
    //AM: Summing RVB in a parallel way
    double R,V,B,somme;
    for ( size_t i = 0; i < local_images_sizes[rank]; i += channels ) {
        R = (double)local_buffer[i];
        V = (double)local_buffer[i+1];
        B = (double)local_buffer[i+2];
        somme = (R + V + B);
        //normalisation des valeurs de pixels dans la plage [-0.5, 0.5] pour ne pas ralentir le processus d'apprentissage
        somme = (somme / 765) - 0.5;
        m_ImageVector.push_back(somme);
    }

    delete image;
}


/*
//AM: Not needed anymore as it is done in a parallel way
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
