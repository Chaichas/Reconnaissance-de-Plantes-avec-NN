#pragma once

#include <iostream>
#include <vector> //vector

//Parameters of Pooling
#define Pooling_size 2 //pooling stride
#define Filter_height 2
#define Filter_width 2 
#define Filter_number 8 //number of filters used in convolution layer


class Pooling_layer {

public:

	Pooling_layer(); //Constructor
	~Pooling_layer(); //Destructor

	void Pooling_parameters(const std::vector<std::vector<double>>& vec_convolution, int input_height, int input_width);

	int getPoolingHeight() { return Pooling_height; } //Return the output height of the pooling matrix
	int getPoolingWidth() { return Pooling_width; } //Return the output width of the pooling matrix

	const std::vector<std::vector<double>>& getPoolingMatrix() { return Pooling_Matrix; } //Return the output pooling 2D matrix

	std::vector<std::vector<double>> BackPropagation(std::vector<std::vector<double>> dloss_dlayer_output); //Backpropagation

	// AM: Moving the Hidden to public to be used after the pre-processing step
	void Hidden(const std::vector<std::vector<double>>& vect); //hidding the last input

private:

	std::vector<std::vector<double>> Pooling_Matrix; //Pooling output matrix
	std::vector<std::vector<double>> HiddenMat_input; //HiddenMat_input matrix

	int HiddenMat_height, HiddenMat_width; //parameters for Hidden

	void Pooling_process(const std::vector<std::vector<double>>& pixel, int idx); //Pooling process

	int Pooling_height, Pooling_width;
	int width, height;
	//AM: Added bool variable for initial resizing of vectors
	bool initialization = true; 
};

