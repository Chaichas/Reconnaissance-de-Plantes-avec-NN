#include "../include/Pooling_layer.h"
#include <algorithm> //for using max elements
#include <cmath>

Pooling_layer::Pooling_layer() {} //constructor


/*Backpropagation from the softmax activation function to the pooling layer

Hidden for implementing the backward phase, crucial for caching:
(1) the input from convolution layer "before flattening" it, (2) the input "after flattening" and (3) the values input of the "softmax activation function" */

void Pooling_layer::Hidden(const std::vector<std::vector<double>>& vect) { //caching

    HiddenMat_input.clear(); //Clear the old HiddenMat_input
    HiddenMat_input.resize(vect.size()); //Resize HiddenMat_input

    //Copy: output.assign(input.begin(), input.end());
    HiddenMat_input.assign(vect.begin(), vect.end()); //output = HiddenMat_input

    HiddenMat_height = (Pooling_height * Pooling_size); //High of the HiddenMat matrix
    HiddenMat_width = (Pooling_width * Pooling_size); //Width of the HiddenMat matrix
}



void Pooling_layer::Pooling_parameters(const std::vector<std::vector<double>>& vec_convolution, int input_height, int input_width) {

    Pooling_height = input_height / Pooling_size; //Height of the output pooling matrix
    Pooling_width = input_width / Pooling_size; //Width of the output pooling matrix

    Pooling_Matrix.clear(); //clear the last matrix

    //Pooling procedure for the 8 filters used
    Pooling_process(vec_convolution, 0); //1st convolution matrix
    Pooling_process(vec_convolution, 1); //2sd convolution matrix
    Pooling_process(vec_convolution, 2); //3rd convolution matrix
    Pooling_process(vec_convolution, 3); //4th convolution matrix
    Pooling_process(vec_convolution, 4); //5th convolution matrix
    Pooling_process(vec_convolution, 5); //6th convolution matrix
    Pooling_process(vec_convolution, 6); //7th convolution matrix
    Pooling_process(vec_convolution, 7); //8th convolution matrix


    Hidden(vec_convolution); //Cache of vec_convolution input
}


//Pooling Process
void Pooling_layer::Pooling_process(const std::vector<std::vector<double>>& pixel, int idx) {

    std::vector<double> vec; //vector vec
     #pragma omp for schedule(dynamic)
     for (int ii = 0; ii < (Pooling_size * Pooling_height); ii += Pooling_size) {
        for (int jj = 0; jj < (Pooling_size * Pooling_width); jj += Pooling_size) {

            std::vector<double> v; //initialize a vector v
            for (int kk = 0; kk < Filter_height; kk++) {
                for (int hh = 0; hh < Filter_width; hh++) {

                    v.push_back(pixel[idx][((ii + kk) * (Pooling_width * Pooling_size) + (jj + hh))]); //store the elements of input matrix to the pooling layer
                }
            }

            //To search the max element, we use the function: max_element(v.begin(), v.end());
            double weightMax = *max_element(v.begin(), v.end()); //search the max of vector v elements
            vec.push_back(weightMax); //store the max weight for each matrix bloc in vec
        }
    }

    Pooling_Matrix.push_back(vec); //The output pooling matrix
}




//In this backpropagation algorithm, we consider the partial derivative of the loss gradient: dloss_dlayer_output ==> Backpropagation from output -> pooling layer -> convolution layer

std::vector<std::vector<double>> Pooling_layer::BackPropagation(std::vector<std::vector<double>> dloss_dlayer_output) { //Backpropagation algorithm for Pooling layer

	//std::vector<std::vector<double>> represents a 2D matrix
	std::vector<std::vector<double>> dloss_dx; //x represents the inputs
        #pragma omp for schedule(dynamic)
	for (size_t ii = 0; ii < Filter_number; ii++) {

		std::vector<double> vec; //initializing vec
		for (int jj = 0; jj < ((Pooling_height * Pooling_size) * (Pooling_width * Pooling_size)); jj++)

			vec.push_back(0); //vec of zeros
		dloss_dx.push_back(vec); //storing vec elements in dloss_dx

	}

	for (int idx = 0; idx < Filter_number; idx++) {

		std::vector<double> vec; //initializing vec
		int iter = 0; //number of iteration

		for (int ii = 0; ii < (Pooling_size * Pooling_height); ii += Pooling_size) {
			for (int jj = 0; jj < (Pooling_size * Pooling_width); jj += Pooling_size) {
				std::vector<double> v;
                                #pragma unroll(16)
				for (int kk = 0; kk < Filter_height; kk++) {
					for (int hh = 0; hh < Filter_width; hh++) {

						v.push_back(HiddenMat_input[idx][(ii + kk) * (Pooling_width * Pooling_size) + (jj + hh)]);
					}
				}

				double weightMax = *max_element(v.begin(), v.end()); //search the max of vector v elements
				bool var = true; //variable var booleen
                                #pragma unroll(16)
				for (int mm = 0; mm < Filter_height; mm++) {
					for (int nn = 0; nn < Filter_width; nn++) {


						//We copy the gradient to the max pixel value
						if (var && (dloss_dlayer_output[idx][iter] == HiddenMat_input[idx][(ii + mm) * (Pooling_width * 2) + (jj + nn)] && weightMax == HiddenMat_input[idx][(ii + mm) * width + (jj + nn)])) {

							dloss_dx[idx][(ii + mm) * (Pooling_width * 2) + (jj + nn)] = weightMax;
							var = false;
						}

						else
							dloss_dx[idx][(ii + mm) * (Pooling_width * Pooling_size) + (jj + nn)] = 0;
					}
				}

				iter++; //incrementing iter
			}
		}
	}

	return dloss_dx; //returning the derivative dloss_dx
}


Pooling_layer::~Pooling_layer()
{

}
