#include "direction_file.h"

#include <dirent.h> //Nous permet de parcourir notre dossier de dataset
#include <algorithm>
#include <memory>
#include <ctime>
#include <iterator>

std::vector<std::string> Reader::Process_directory(const std::string& directory, std::vector<int>& label) {

    DIR*    dir;
    dirent* pDir;
    std::vector<std::string> vect_files;
    
    //opendir() nous permet d'Ouvrir le répertoire et retourne un pointeur --> dir
    dir = opendir(directory.c_str());
    
    //Si le dossier n'a pas pu être ouvert 
    if(NULL == dir){
        std::cout << "could not open directory: " <<std::endl;
        return;
    }
    
    /*readdir() nous permet de lire ce qui est dedans le repertoire, il prend en paramètre un pointeur sur DIR.
    */
    pDir = readdir(dir)
    while (pDir != NULL) {
      std::string name = pDir->d_name; //d_name contient le nom de notre fichier
      auto place = name.find(".");
      std::string extension = name.substr(place + 1);
      if (extension == "jpg") {
        std::string path= directory+"/"+name;
        temp_files.push_back(path);
        //Get label
        char l = name[0];
        int l2 = l-'0';
        label.push_back(l2);
      }
      pDir = readdir(dir)
    }
    return vect_files;
}

/*
     Reference : - https://pub.phyks.me/sdz/sdz/arcourir-les-dossiers-avec-dirent-h.html (Tutoriel : Parcourir les dossiers avec dirent.h)
*/
