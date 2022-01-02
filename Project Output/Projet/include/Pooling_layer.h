#ifndef __POOLING_LAYER_H__
#define __POOLING_LAYER_H__

#include <iostream>
#include <vector> //vector


class Pooling_layer{

public:

	Pooling_layer(); //Constructor
	~Pooling_layer(); //Destructor
	
	void Pooling_parameters(const std::vector<double>& vec_convolution, int input_height, int input_width);
	
	int getPoolingHeight() {return Pooling_height;} //Return the output height of the pooling matrix
	int getPoolingWidth() {return Pooling_Width;} //Return the output width of the pooling matrix
	
	const std::vector<std::vector<double>>& getPoolingMatrix() {return Pooling_Matrix;} //Return the output pooling 2D matrix
	
	
private:

std::vector<std::vector<double>> Pooling_Matrix; //Pooling output matrix

void Pooling_process(const std::vector<double>& pixel, int idx); //Pooling process



};

#endif
