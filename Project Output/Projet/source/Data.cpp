#include <iostream>
#include <string>
#include "Data.h"

Data::Data(){}
Data::~Data(){}

void Data::loadImage(const std::string& str_Path, int& Hauteur, int& largeur)
{
  m_ImageVector.clear();
 
  cv::Mat image = cv::imread( str_Path, CV_LOAD_IMAGE_COLOR);
  
  if(!image.data)
  {
        std::cout << "Error: the image wasn't correctly loaded." << std::endl;
        return -1;
  }

  hauteur = image.rows;
  largeur = image.cols;

  this->create_canal(image);

}

void Data::create_canal(cv::Mat image)
{
  int i=0;
  while (i<image.rows)
  {
      int j=0;
      while(j<image.cols)
      {
        double R = ((double)image.at<cv::Vec3b>(i,j)[2]);
        double V = ((double)image.at<cv::Vec3b>(i,j)[1]);
        double B = ((double)image.<cv::Vec3b>(i,j)[0]);
        double somme = (R+V+B);
        somme = (somme/765)-0.5;
        m_ImageVector.push_back(somme);
        j++;
      }
      i++;
  }
}


