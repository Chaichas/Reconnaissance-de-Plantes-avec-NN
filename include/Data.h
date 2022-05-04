#pragma once

#include <vector>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
using namespace cv;

class Data
{
private:
	
	void create_canal(Mat* image);

public:
	Data();
	~Data();
	std::vector<double> m_ImageVector;
	const std::vector<double>& get_fusion_canal() const;
	void loadImage(const std::string& My_path, int& hauteur, int& largeur);
};
