#pragma once

#include <iostream>
#include "Data.h"
#include<vector>

#include <iostream>
#include <vector> //vector
#include<mpi.h>
#include <chrono> //time
#include <cmath> //math
#include <random> //random

//Input Parameters for the Convolution_layer class
#define filter_number 8  //number of filters used
#define filter_height 3 //hight of the filter
#define filter_width 3 //width of the filter
#define padding 0 //padding (explained in the descriptive)
#define stride_conv 1 //offset
#define pooling_size 2 //pooling size

//Note: This is the case of a valid convolution (i.e., the padding is null)

class Convolution_layer {

public:
	Convolution_layer(); //Constructor
	~Convolution_layer(); //Destructor

	void convolution_parameters(const std::vector<double>& vec_pixel, int inputImage_height, int inputImage_width);

	//Get Height and Width of the convolutional output matrix
	int getMatHeight() { return ConvMat_height; } //Convolution output matrix height
	int getMatWidth() { return ConvMat_width; } //Convolution output matrix width

	//Get Convolution output Matrix
	const std::vector<std::vector<double>>& getConvMat() const { return ConvMat; }
	void static random_weights(double nb_filters, double nb_weights, std::vector<std::vector<double>>& filter_matrix);


private:

	std::vector<std::vector<double>> ConvMat; //convolution matrix filled with values
	std::vector<std::vector<double>> proc_ConvMat; // temporary convolution matrix filled by only one proc
	
	std::vector<std::vector<double>> filter_matrix; //Filter matrix filled with values

	int ConvMat_height, ConvMat_width; //, proc_ConvMat_height; //con matrix intialization

	void convolution_process(const std::vector<double>& pixel, int idx, const int ls_rank, const int le_rank); //multiplication and summation pixel value * filter value

	//Backpropagation: Hiding inputs
	std::vector<double> HiddenMat; //Hidding of inputs 
	void Hidden(const std::vector<double>& vect);
	bool initialization = true; //initializing filters with random values 

public:

	//Case of Concolution layer propagation
	void BackPropagation(std::vector<std::vector<double>> dloss_dlayer_output, double LearningRate);

};

