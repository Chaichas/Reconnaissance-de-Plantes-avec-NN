#pragma once

#include <vector>
#include "Data.h"
#include "Convolution_layer.h"
#include "Pooling_layer.h"
#include "softmax_layer.h"

class output
{

private:
	//Dataset pour le training et le test
   // std::string file_train;
	//std::string file_test;

	std::string  m_trainPath;
	std::string m_testPath;

	Data* m_image;
	Convolution_layer* m_convol;
	Pooling_layer* m_pool;
	Softmax_layer* m_softmax;
	void train(int label, int& hauteur, int& largeur, double& lRate);
	////pour recuperer le vecteur de probabilte de sortie 
	std::vector<double> prediction(int label, int& hauteur, int& largeur);
	double acc = 0.0;
	double loss = 0.0;
public:
	output(std::string file_train, std::string file_test);
	~output();

	void Training_data(int train_epoch = 400, double alpha = 0.005);
	void Testing_data();
	std::vector<std::string> Process_directory(const std::string& path, std::vector<int>& label);
};

