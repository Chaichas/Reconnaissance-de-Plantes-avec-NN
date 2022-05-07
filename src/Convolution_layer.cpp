#include <iostream>
#include "../include/Convolution_layer.h"
#include<vector>
#include <mpi.h>

Convolution_layer::Convolution_layer() {} //constructor


/*Here, we will update the weights of the filters using the Backpropagation

Hidden contain the hidden layers in the backward phase, crucial for caching:
(1) the input from convolution layer "before flattening" it, (2) the input "after flattening" and (3) the values input of the "softmax activation function" */

void Convolution_layer::Hidden(const std::vector<double>& vect) { //caching

    //HiddenMat.clear(); //Clear the old HiddenMat
    //HiddenMat.resize(vect.size()); //Resize HiddenMat

    //Copy: output.assign(input.begin(), input.end());
    HiddenMat.assign(vect.begin(), vect.end()); //output = HiddenMat
}


//in convolution we use the whole volume of the input matrix n*n*channels(RGB, 3)
void Convolution_layer::convolution_parameters(const std::vector<double>& vec_pixel, int inputImage_height, int inputImage_width) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Comm_size(MPI_COMM_WORLD,&size);

    ConvMat_height = ((inputImage_height - filter_height + 2 * padding) / stride_conv) + 1; //output convolution matrix height
    ConvMat_width = ((inputImage_width - filter_width + 2 * padding) / stride_conv) + 1; //output convolution matrix width

    //initialization of the filter weights by random values (class Random_weights)
    //fprintf(stderr,"Hi");
	if (initialization) {
        HiddenMat.resize(vec_pixel.size());
        ConvMat.resize(filter_number, std::vector<double> (ConvMat_height*ConvMat_width));
        filter_matrix.resize(filter_number, std::vector<double> (filter_height*filter_width));
        std::vector<double> simplified_filter (filter_number*filter_height*filter_width);
        if(rank==0) { // le filtre 
			random_weights(filter_number, filter_height * filter_width, filter_matrix); //initialization of weights with random values

			//normalizing wight values, Ref3
			for (size_t ii = 0; ii < filter_number; ii++) { //loop on number of filters
				for (size_t jj = 0; jj < (filter_height * filter_width); jj++) { //loop on total number of weights within each filter
					filter_matrix[ii][jj] = (double)filter_matrix[ii][jj] / (double)(filter_height * filter_width); //normalized filter
				}
			}
            // transformation de filter_matrix en un vecteur 1d pour un partage plus facile
            size_t idx = 0;
            for (size_t ii = 0; ii < filter_number; ii++) {
                for (size_t jj = 0; jj < (filter_height * filter_width); jj++) {
                    simplified_filter[idx] = filter_matrix[ii][jj];
                    idx++;
                }
            }
		}

        MPI_Bcast(&simplified_filter[0],filter_number*filter_height*filter_width,MPI_DOUBLE,0,MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);
        //std::fprintf(stderr,"Iam rank: blala %d, %ld \n",rank,simplified_filter.size());
        // retransformation du vecteur en matrice de filtre
        for (size_t ii = 0; ii < filter_number; ii++) {
            for (size_t jj = 0; jj < (filter_height * filter_width); jj++) {
                filter_matrix[ii][jj] = simplified_filter[ii*filter_height*filter_width+jj];
            }
        }
        initialization = false;
        regionsBack.resize(ConvMat_height*ConvMat_width, std::vector<double> (filter_height*filter_width));
	}
    //fprintf(stderr,"Hello");

    //ConvMat.clear();
    

    //Convolution procedure for filter_number
    convolution_process(vec_pixel, 0); //1st filter
    convolution_process(vec_pixel, 1); //2sd filter
    convolution_process(vec_pixel, 2); //3rd filter
    convolution_process(vec_pixel, 3); //4th filter
    convolution_process(vec_pixel, 4); //5th filter
    convolution_process(vec_pixel, 5); //6th filter
    convolution_process(vec_pixel, 6); //7th filter
    convolution_process(vec_pixel, 7); //8th filter
    //fprintf(stderr,"Hey");
    Hidden(vec_pixel); //hiding the last input
    //fprintf(stderr,"How are you ");
}

void Convolution_layer::convolution_process(const std::vector<double>& pixel, int idx) {
    //std::vector<double> vec;
    int count = 0;
    for (int ii = 0; ii < ConvMat_height; ii++) { //loop on the height of the convolution matrix
        for (int jj = 0; jj < ConvMat_width; jj++) { //loop on the width of the convolution matrix

            double sum = 0; //initialization of the summation

            for (int kk = 0; kk < filter_height; kk++) { //loop on the height of the filter
                for (int hh = 0; hh < filter_width; hh++) { //loop on the width of the filter

                    double image = (pixel[((ii + kk) * (ConvMat_width + 2) + (jj + hh))]); //pixel value of the image stored in image

                    sum += (image * filter_matrix[idx][kk * filter_width + hh]); //sum of the pixel value * filter value
                }

                
            }
            //vec.push_back(sum); //storing sum value in vec	AM	
            ConvMat[idx][count] = sum;
            count++;
        }
    }
    //ConvMat.push_back(vec); //storing vec value in ConvMat  AM
}

void Convolution_layer::BackPropagation(std::vector<std::vector<double>> d_L_d_out, double learn_rate)
{
    //d_L_d_out is the loss gradient for this layer's outputs
    //filters with same shape as filter_matrix
    std::vector<std::vector<double>> filters (filter_number, std::vector<double> (filter_height * filter_width, 0.0));
    /*for (size_t i = 0; i < filter_number; i++) {
        std::vector<double> v;
        for (int j = 0; j < (filter_height * filter_width); j++)
            //v.push_back(0); //AM
            v[j] = 0; //AM
        //filters.push_back(v); //AM
        filters[i] = v; //AM
    }*/

    //For keeping 3x3 reegions of last input
    //std::vector<std::vector<double>> regions (ConvMat_height*ConvMat_width, std::vector<double> (filter_height*filter_width));

    //Loop for storing 3x3 regions into "regions"
    int idx1, idx2;
    idx1 = 0;
    for (int i = 0; i < ConvMat_height; i++)
    {
        for (int j = 0; j < ConvMat_width; j++) {
            //double sum = 0;
            //std::vector<double> v;
            idx2 = 0;
            for (int k = 0; k < filter_height; k++) {
                for (int n = 0; n < filter_width; n++) {
                    //v.push_back(HiddenMat[((i + k) * (ConvMat_width + 2) + (j + n))]); 
                    regionsBack[idx1][idx2] = HiddenMat[((i + k) * (ConvMat_width + 2) + (j + n))];
                    idx2++;
                }
            }
            //regions.push_back(v); //AM
            //regions[j] = v;
            idx1++;
        }
    }

    //Loop for iterating d_L_d_out(last output of this layer)
    int counter = 0;
    for (int i = 0; i < ConvMat_height; i++) {
        for (int j = 0; j < ConvMat_width; j++) {
            //Loop for filters number
            for (size_t k = 0; k < filter_number; k++) {
                for (size_t m = 0; m < 3; m++) {
                    for (size_t n = 0; n < 3; n++) {
                        filters[k][m * 3 + n] += ((d_L_d_out[k][i * ConvMat_width + j] * regionsBack[counter][m * 3 + n]));
                    }
                }
            }
            counter++;
        }
    }

    for (size_t i = 0; i < filter_number; i++) {
        for (size_t j = 0; j < 3; j++) {
            for (size_t k = 0; k < 3; k++) {
                filter_matrix[i][j * 3 + k] -= (learn_rate * filters[i][j * 3 + k]);
            }
        }
    }
}
void Convolution_layer::random_weights(double nb_filters, double nb_weights, std::vector<std::vector<double>>& filter_matrix) {

    //construct a random generator engine from a time-based seed, Ref1
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count(); //time; system real time

    std::default_random_engine generator(seed); //random generator engine
    std::normal_distribution<double> distribution(0.0, 1.0); //( result_type mean = 0.0, result_type stddev = 1.0 )

    for (int ii = 0; ii < nb_filters; ii++) { //loop on the total number of filters

        //std::vector<double> one_filter (nb_weights); //array initialisation with no defined size

        for (int jj = 0; jj < nb_weights; jj++) { //nb_weights = filter height * filter width

            double number = (distribution(generator)); //random number from a random generator engine, Ref2

            filter_matrix[ii][jj] = number; // Filling of 1 filter with random values

        }

        //filter_matrix[ii] = one_filter; //Filling of all filters with random values
    }

}
Convolution_layer::~Convolution_layer()
{

}
/*	References:

    Ref1: https://www.cplusplus.com/reference/random/normal_distribution/normal_distirbution/

    Ref2: https://www.cplusplus.com/reference/random/normal_distribution/

 */