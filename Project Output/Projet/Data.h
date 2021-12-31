#pragma once

#include <vector>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

class Data
{
private:
  std::vector<double> m_ImageVector;
  void crate_canal(cv::Mat image);

public:
  Data();
  ~Data();
  void loadImage(const std::string& My_path, int& hauteur, int& largeur);
};
