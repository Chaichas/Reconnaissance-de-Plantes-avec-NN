
//-------------------------------------------Training Part------------------------------------------------------

void Output::Training_data(int numb_epoch, double alpha)
{
  std::cout << "--------------Start Training ----------" << '\n';
  
  std::vector<int> Labels;
  std::vector<std::string> training_files = Reader::Process_directory(m_trainPath, Labels);

  int it=0;
  while(it < numb_epoch) {

    int label_i = 0;
    double runningAcc = 0.0, runningLoss = 0.0;

    for (std::vector<std::string>::iterator it = training_files.begin() ; it != training_files.end(); ++it)
    {
      std::string name = *it;
      int hauteur = 0, largeur = 0;
      
      m_image->loadImage(name, hauteur, largeur);
      label_i = std::distance(trainingFiles.begin(), it);
           
      //Lancement de training
      train(Labels[label_i], hauteur, largeur, alpha);

      runningLoss += loss;
      runningAcc += acc;
    }
    
    label_i++;
    
    //Affichage de perte et de la precison pour chaque epoch 
    std::cout << "Epoch " << it << " : Average Loss " << runningLoss / label_i << " , Accuracy " << (runningAcc/label_i)*100 << " %" << '\n';
    runningLoss = 0.0;
    runningAcc = 0.0;
    it++;
  }
}

//-------------------------------------------Testing Part------------------------------------------------------

void Output::Testing_data()
{
  int label_i = 0;
  double runningAcc = 0.0, runningLoss = 0.0;

  std::cout << "-----------------Testing Data -------------------" << '\n';

  std::vector<int> labels_test;
  std::vector<std::string> testing_files = Reader::Process_directory(m_trainPath, Labels);

  int right = 0;

  for (std::vector<std::string>::iterator it = testingFiles.begin() ; it != testingFiles.end(); ++it)
  {
    std::string name = *it;
    int hauteur = 0, largeur = 0;
    
    m_image->loadImage(name, hauteur, largeur);
    label_i = std::distance(testingFiles.begin(), it);
    
    //Recuperation de vecteur de probabilté de sortie 
    std::vector<double> out = prediction(labels_test[labelIndex], height, width);

    // TODO function out vector of prediction 

    runningAcc += m_loss;
    runningLoss += m_acc;

    int predIndex = std::distance(out.begin(), std::max_element(out.begin(), out.end()));
    if(labels_test[label_i]==predIndex)
      right++;

  }
  label_i++;
  
  int wrong = label_i-right;
  std::cout<<"--------------------------------Result of testing-------------------------"<< '\n';
  
  std::cout << "Average Loss" << runningLoss / label_i << " , Accuracy "<< '\n';
  std::cout << "Accuracy "<< (runningAcc/label_i)*100 << " %." << '\n';
  
  std::cout<<"---------------------------------------------------------------------------"<< '\n';
  std::cout << "Le nombre d'image de test est : "<< label_i << '\n';
  std::cout << "Le nombre d'image correctement prédits : "<< right <<'\n';
  std::cout << "Le nombre d'image non prédits : "<< wrong <<'\n';




