#Plants Recognition using CNN, 2 classes
#Reference: https://www.analyticsvidhya.com/blog/2020/10/create-image-classification-model-python-keras/

#Import the required libraries 
import matplotlib.pyplot as plt #Visualization of graphs
import seaborn as sns #static data visualization

import keras #Generate training and test models
from keras.models import Sequential #Each layer has one input tensor and one output tensor

#Dense: the regular deeply connected neural network layer
#Conv2D: the most common type of convolution that is used (The 2D convolution layer)
#MaxPool2D: used to reduce the size of the image by the maximum value within a matrix
#Flatten: turns a matrix to a one dimensional array  
#Dropout: A regularization technique when we have a reduced training database
from keras.layers import Dense, Conv2D , MaxPool2D , Flatten , Dropout

from keras.preprocessing.image import ImageDataGenerator #Real-time data augmentation
from tensorflow.keras.optimizers import Adam #Optimizer that implements the Adam algorithm (stochastic gradient descent)

#classification_report: Builds a text report showing the main classification metrics
#confusion_matrix: evaluates the accuracy of a classification
from sklearn.metrics import classification_report,confusion_matrix

import tensorflow as tf #fast numerical computing for ML
import cv2 #Image Processing
import os #creating and removing a directory (folder)

import numpy as np #working with arrays


# Model configuration
labels = ['Chlorophyta', 'Liliopsida'] #Classes Names
img_size = 224 #Images Size
#Loading the data from the Database Directory
def get_data(data_dir): 
    data = [] 
    for label in labels: 
        path = os.path.join(data_dir, label)
        class_num = labels.index(label)
        for img in os.listdir(path):
            try:
                img_arr = cv2.imread(os.path.join(path, img))[...,::-1] #convert BGR to RGB format
                resized_arr = cv2.resize(img_arr, (img_size, img_size)) # Reshaping images to preferred size
                data.append([resized_arr, class_num])\
            except Exception as e:
                print(e)
    return np.array(data)
#Now we can easily fetch our train and validation data using the paths
train = get_data('/home/aicha/Downloads/aysha-20211128T121956Z-001/aysha/train')
val = get_data('/home/aicha/Downloads/aysha-20211128T121956Z-001/aysha/test')

#Visualize the data (output: bars of classes in function of the number of images, using Seaborn)
l = []
for i in train:
    if(i[1] == 0):
        l.append("Chlorophyta")
    else:
        l.append("Liliopsida")
sns.set_style('darkgrid')
sns.countplot(l)

#visualize a random image
#plt.figure(figsize = (5,5))
#plt.imshow(train[1][0])
#plt.title(labels[train[0][1]])

#visualize a random image
#plt.figure(figsize = (5,5))
#plt.imshow(train[-1][0])
#plt.title(labels[train[-1][1]])

#Data Preprocessing and Data Augmentation before we can proceed with building the model
x_train = []
y_train = []
x_val = []
y_val = []

for feature, label in train:
  x_train.append(feature)
  y_train.append(label)

for feature, label in val:
  x_val.append(feature)
  y_val.append(label)

# Normalize the data
x_train = np.array(x_train) / 255
x_val = np.array(x_val) / 255
x_train.reshape(-1, img_size, img_size, 1)
y_train = np.array(y_train)
x_val.reshape(-1, img_size, img_size, 1)
y_val = np.array(y_val)

#Data augmentation on the train data
datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range = 30,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.2, # Randomly zoom image 
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip = True,  # randomly flip images
        vertical_flip=False)  # randomly flip images


datagen.fit(x_train)

#Define the Model
model = Sequential()
model.add(Conv2D(32,3,padding="same", activation="relu", input_shape=(224,224,3))) #3 Convolutional layers
model.add(MaxPool2D()) #Definition of max-pooling layers

model.add(Conv2D(32, 3, padding="same", activation="relu"))
model.add(MaxPool2D())

model.add(Conv2D(64, 3, padding="same", activation="relu"))
model.add(MaxPool2D())
model.add(Dropout(0.4)) #A dropout layer is added after the 3rd maxpool operation to avoid the overfitting

model.add(Flatten())
model.add(Dense(128,activation="relu"))
model.add(Dense(2, activation="softmax"))

model.summary()

##Adam: optimizer and SparseCategoricalCrossentropy: the loss function
opt = Adam(lr=0.000001) #lower learning rate of 0.000001 for a smoother curve
model.compile(optimizer = opt , loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True) , metrics = ['accuracy'])

##Model Training for 500 epochs since the learning rate is very small
history = model.fit(x_train,y_train,epochs = 500 , validation_data = (x_val, y_val))

#Evaluating the result
#Plot the training and validation accuracy along with the training and validation loss
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(500) #parameter of graph

#Plotting
plt.figure(figsize=(15, 15))
plt.subplot(2, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(2, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

#We can print out the classification report to see the precision and accuracy
#predictions = model.predict_classes(x_val)
#predictions = predictions.reshape(1,-1)[0]
#print(classification_report(y_val, predictions, target_names = ['Chlorophyta (Class 0)','Liliopsida (Class 1)']))
