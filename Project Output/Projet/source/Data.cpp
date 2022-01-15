#include <iostream>
#include <string>
#include "image.h"

Data::Data() {}
Data::~Data() {}

void Data::loadImage(const std::string& str_Path, int& hauteur, int& largeur)
{
    //Recuperation de l'image

    m_ImageVector.clear();
    Mat* image = new Mat();
    //Creation et lecture de l'image
    (*image) = imread(str_Path, IMREAD_COLOR);
    /*
    if(!image.data)
    {
          std::cout << "Error: the image wasn't correctly loaded." << std::endl;
          return -1;
    }
    */
    //Recuperation des dimension de l'image
    hauteur = image->rows;
    largeur = image->cols;

    //Generation de vecteur RVB
    this->create_canal(image);
    delete image;
}

void Data::create_canal(Mat* image)
{
    int i = 0;
    while (i < image->rows)
    {
        int j = 0;
        while (j < image->cols)
        {
            double R = ((double)image->at<Vec3b>(i, j)[2]);
            double V = ((double)image->at<Vec3b>(i, j)[1]);
            double B = ((double)image->at<Vec3b>(i, j)[0]);
            double somme = (R + V + B);

            //normalisation des valeurs de pixels dans la plage [-0.5, 0.5] pour ne pas ralentir le processus d'apprentissage
            somme = (somme / 765) - 0.5;
            m_ImageVector.push_back(somme);
            j++;
        }
        i++;
    }
}

const std::vector<double>& Data::get_fusion_canal() const
{
    return m_ImageVector;
}

/*
     Reference : - https://machinelearningmastery.com/how-to-manually-scale-image-pixel-data-for-deep-learning/
                 - https://sodocumentation.net/opencv/topic/1957/pixel-access
*/
