#include <iostream>
#include "../include/Convolution_layer.h"
#include<vector>

Convolution_layer::Convolution_layer() {} //constructor


/*Here, we will update the weights of the filters using the Backpropagation

Hidden contain the hidden layers in the backward phase, crucial for caching:
(1) the input from convolution layer "before flattening" it, (2) the input "after flattening" and (3) the values input of the "softmax activation function" */

void Convolution_layer::Hidden(const std::vector<double>& vect) { //caching

    HiddenMat.clear(); //Clear the old HiddenMat
    HiddenMat.resize(vect.size()); //Resize HiddenMat

    //Copy: output.assign(input.begin(), input.end());
    HiddenMat.assign(vect.begin(), vect.end()); //output = HiddenMat
}


//in convolution we use the whole volume of the input matrix n*n*channels(RGB, 3)
void Convolution_layer::convolution_parameters(const std::vector<double>& vec_pixel, int inputImage_height, int inputImage_width) {

    ConvMat_height = ((inputImage_height - filter_height + 2 * padding) / stride_conv) + 1; //output convolution matrix height
    ConvMat_width = ((inputImage_width - filter_width + 2 * padding) / stride_conv) + 1; //output convolution matrix width

    //AM: Getting the MPI rank and size
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Comm_size(MPI_COMM_WORLD,&size);
    
    filter_matrix.resize(filter_number, std::vector<double> (filter_height*filter_width));
    //AM: The simplified_filter is used for easier MPI data communication
    std::vector<double> simplified_filter (filter_number*filter_height*filter_width);
	if(rank==0) { // le filtre 
    //initialization of the filter weights by random values (class Random_weights)
		if (initialization) {
			random_weights(filter_number, filter_height * filter_width, filter_matrix); //initialization of weights with random values

			//normalizing wight values, Ref3
			for (size_t ii = 0; ii < filter_number; ii++) { //loop on number of filters
				for (size_t jj = 0; jj < (filter_height * filter_width); jj++) { //loop on total number of weights within each filter
					filter_matrix[ii][jj] = (double)filter_matrix[ii][jj] / (double)(filter_height * filter_width); //normalized filter
				}
			}
			initialization = false;
		}
		// AM: transformation of filter_matrix to a 1D vector for easier data sharing in MPI
		size_t idx = 0;
		for (size_t ii = 0; ii < filter_number; ii++) {
			for (size_t jj = 0; jj < (filter_height * filter_width); jj++) {
				simplified_filter[idx] = filter_matrix[ii][jj];
                idx++;
			}
		}
	}
    // AM: Brodcasting the filter to other procs
	MPI_Bcast(&simplified_filter[0],filter_number*filter_height*filter_width,MPI_DOUBLE,0,MPI_COMM_WORLD);
	
    // AM: retransformation of the vector to a filter matrix
	for (size_t ii = 0; ii < filter_number; ii++) {
		for (size_t jj = 0; jj < (filter_height * filter_width); jj++) {
			filter_matrix[ii][jj] = simplified_filter[ii*filter_height*filter_width+jj];
		}
	}

    ConvMat.clear();
    ConvMat.resize(filter_number, std::vector<double> (ConvMat_height*ConvMat_width));

    // AM: simplified_filter is the computed convolution matrix
    proc_ConvMat.clear();

	// AM: line start and line end for each proc. They star in 1 because the filter needs an additional row
	int ls_rank = 1;
	int le_rank = 1;
	//AM: optimization of the repartition of the image row between the procs (load distribution)
	if (rank < ConvMat_height%size)
	{
		ls_rank += (ConvMat_height/size)*rank + rank;
		le_rank += (ConvMat_height/size)*(rank+1) + rank + 1;
	}
	else
	{
		ls_rank += (ConvMat_height/size)*rank + ConvMat_height%size;
		le_rank += (ConvMat_height/size)*(rank+1) + ConvMat_height%size;
	}

	int proc_ConvMat_height = le_rank-ls_rank; 
    //AM: We add 1 to the end row, with was originally substracted from the start row, necessary for the filter 
    le_rank++;
    ls_rank--;
	//std::vector<double> vec_pixel_rank (inputImage_width*(proc_ConvMat_height+2));
    //vec_pixel_rank.assign(vec_pixel.begin()+inputImage_width*(ls_rank-1),vec_pixel.begin()+inputImage_width*(le_rank+1));
	//std::copy(&vec_pixel[inputImage_width*(ls_rank-1)], &vec_pixel[inputImage_width*(le_rank+1)], vec_pixel_rank.begin());
	
    //Convolution procedure for filter_number
	for (size_t ii = 0; ii < filter_number; ii++) {
		convolution_process(vec_pixel, ii, ls_rank, le_rank);
	}
	
    //AM: Computing counts and displ needed for gatherv
	int counts[size];
	int displ[size];
    std::vector<double> global_vec; //AM: used to gather all data at once
    if (rank == 0) {
        displ[0] = 0;
        for (int i_rank=0; i_rank<size; i_rank++) {
            if (i_rank < ConvMat_height%size) {
                counts[i_rank] = (ConvMat_height/size)+1;
            }
            else {
                counts[i_rank] = ConvMat_height/size;
            }
            counts[i_rank] *= ConvMat_width*filter_number;
            if (i_rank > 0) displ[i_rank] = displ[i_rank-1]+counts[i_rank-1];
        }
        
        global_vec.resize(ConvMat_height*ConvMat_width*filter_number);
    }
    //AM: The simplified filter is used again for easier MPI communication sharing
	simplified_filter.resize(proc_ConvMat_height*ConvMat_width*filter_number);
	int idx = 0;
    for (int ii = 0; ii < filter_number; ii++) {
        for (int jj = 0; jj < proc_ConvMat_height*ConvMat_width; jj++) {
            simplified_filter[idx] = proc_ConvMat[ii][jj];
            idx++;
        }
    }
    
    //AM: Check that all processors are ready
    MPI_Barrier(MPI_COMM_WORLD);
    //AM: Gather all computed convolution matrices by the processors in rank 0. The gathered global vec contains the 8 filters (Approach 1bis)
	MPI_Gatherv(&simplified_filter[0], proc_ConvMat_height*ConvMat_width*filter_number, MPI_DOUBLE, &global_vec[0], counts, displ, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	if (rank == 0) {
        //AM: After receiving all the data, rank 0 will rearrange them and store them in ConvMat
        for (int ii = 0; ii < filter_number; ii++) {
            idx = 0;
            for (int i_rank=0; i_rank<size; i_rank++) {
                int nbr_per_filter = counts[i_rank]/filter_number;
                //fprintf(stderr,"nbr_per_filter %d \n",nbr_per_filter);
                int nbr_per_filter_ii = ii * nbr_per_filter;
                for (int jj=0; jj<nbr_per_filter; jj++) {
                    ConvMat[ii][idx] = global_vec[displ[i_rank]+nbr_per_filter_ii+jj];
                    idx++;
                }
            }
        }
    }
    //fprintf(stderr,"size %d \n",vec_pixel.size());
    //AM: Approach 1 before modification to approach 1bis. We gather the data for each filter without mixing them
	/*MPI_Barrier(MPI_COMM_WORLD);
	//std::vector<double> simplified_proc_ConvMat (ConvMat_width*proc_ConvMat_height*filter_number);
	//std::vector<double> global_vec (ConvMat_height*ConvMat_width);
	for (size_t ii = 0; ii < filter_number; ii++) {
		//std::copy(&proc_ConvMat[ii][0], &proc_ConvMat[ii][ConvMat_width*proc_ConvMat_height], filter_proc_ConvMat.begin());
        //std::fprintf(stderr,"I am rank %d, counts : %d, displ : %d, proc_ConvMat_height : %d\n",rank,counts[0],displ[0],proc_ConvMat_height);
        MPI_Gatherv(&proc_ConvMat[ii][0], ConvMat_width*proc_ConvMat_height, MPI_DOUBLE, &ConvMat[ii][0], counts, displ, MPI_DOUBLE, 0, MPI_COMM_WORLD);
		//MPI_Allgatherv(&filter_proc_ConvMat[0], filter_proc_ConvMat.size(), MPI_DOUBLE, &global_vec[0], counts, displ, MPI_DOUBLE, MPI_COMM_WORLD);
		//ConvMat.push_back(global_vec);
	}*/
    MPI_Barrier(MPI_COMM_WORLD);

    Hidden(vec_pixel); //hiding the last input
}

void Convolution_layer::convolution_process(const std::vector<double>& pixel, int idx, const int ls_rank, const int le_rank) {
    std::vector<double> vec;

    for (int ii = ls_rank; ii < le_rank; ii++) { //loop on the height of the convolution matrix
        for (int jj = 0; jj < ConvMat_width; jj++) { //loop on the width of the convolution matrix

            double sum = 0; //initialization of the summation

            for (int kk = 0; kk < filter_height; kk++) { //loop on the height of the filter
                for (int hh = 0; hh < filter_width; hh++) { //loop on the width of the filter

                    double image = (pixel[((ii + kk) * (ConvMat_width + 2) + (jj + hh))]); //pixel value of the image stored in image

                    sum += (image * filter_matrix[idx][kk * filter_width + hh]); //sum of the pixel value * filter value

                }
            }
            vec.push_back(sum); //storing sum value in vec		
        }
    }
    proc_ConvMat.push_back(vec); //storing vec value in ConvMat
}

void Convolution_layer::BackPropagation(std::vector<std::vector<double>> d_L_d_out, double learn_rate)
{
    //d_L_d_out is the loss gradient for this layer's outputs
    //filters with same shape as filter_matrix
    std::vector<std::vector<double>> filters;
    for (size_t i = 0; i < filter_number; i++) {
        std::vector<double> v;
        for (int j = 0; j < (filter_height * filter_width); j++)
            v.push_back(0);
        filters.push_back(v);
    }

    //For keeping 3x3 reegions of last input
    std::vector<std::vector<double>> regions;

    //Loop for storing 3x3 regions into "regions"
    for (int i = 0; i < ConvMat_height; i++)
    {
        for (int j = 0; j < ConvMat_width; j++) {
            //double sum = 0;
            std::vector<double> v;
            for (int k = 0; k < filter_height; k++) {
                for (int n = 0; n < filter_width; n++) {
                    v.push_back(HiddenMat[((i + k) * (ConvMat_width + 2) + (j + n))]);
                }
            }
            regions.push_back(v);
        }
    }

    //Loop for iterating d_L_d_out(last output of this layer)
    int counter = 0;
    for (int i = 0; i < ConvMat_height; i++) {
        for (int j = 0; j < ConvMat_width; j++) {
            //Loop for filters number
            for (size_t k = 0; k < 2; k++) {
                for (size_t m = 0; m < 3; m++) {
                    for (size_t n = 0; n < 3; n++) {
                        filters[k][m * 3 + n] += ((d_L_d_out[k][i * ConvMat_width + j] * regions[counter][m * 3 + n]));
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