#pragma once

#include <vector>
#include "Data.h"
#include "convolution_layer.h"
#include "pooling.h"
#include "softmax.h"

class Output
{

private:
  //Dataset pour le training et le test
  std::string file_train;
  std::string file_test;

  Data * m_image;
  Convolution_layer * m_convol;
  Pooling * m_pool;
  Softmax * m_softmax;
  void train(int label, int& hauteur, int& largeur, double& lRate);
  ////pour recuperer le vecteur de probabilté de sortie 
  std::vector<double> prediction(int label, int& hauteur, int& largeur);

public:
  output(std::string file_train, std::string file_test);
  ~output();

  void Training_data(int train_epoch = 400, double alpha = 0.005);
  void Testing_data();
};


