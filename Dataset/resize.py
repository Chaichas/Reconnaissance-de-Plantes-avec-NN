import cv2
import os
import sys

#----------------------------
#img_path = "rose.jpg"
#img = cv2.imread(img_path)
#width = 200
#height = 200
#---------------------------

#resize images function 
def resize_img(img_path,size): #Path + New size
    #print("sizeeeeeee:",size)
    width = size
    #height = 50
    img = cv2.imread(img_path) #read images 
    print("Original size: ",img.shape)
    new_img = cv2.resize(img,(width,width)) #resize image with new dimesion 
    #new_img = cv2.resize(img,(size,size))
    print("New size: ",new_img.shape)

    return new_img
    
#-------------------------------------------
#print("Original size: ",img.shape)
#new_img = cv2.resize(img,(width,height))
#print("New size: ",new_img.shape)
#cv2.imshow("Original",img)
#cv2.imshow("New size",new_img)
#cv2.waitKey(0)
#-------------------------------------------

in_dir = str(input("in dir path ?: "))
out_dir = str(input("Out dir name ?: "))
size = int(input("New images size ?: "))
number = int(input("Number of images in class ?: "))

#create output dir if not exist 
if not os.path.exists(out_dir):
    os.makedirs(out_dir)


class_count = 0
image_count = 0

for tsfn in os.listdir(in_dir):  #tesf = class / get class from indir

    #classPath = imgs_path + '\\' + tsfn
    classPath = os.path.join(in_dir,tsfn) #storge path class=tesf in classPath
    print(classPath)
    if os.path.isdir(classPath):

        print("Processing class: ", class_count) 
        for img in os.listdir(classPath):
            
            if number == image_count:  #check max image = pour la definition de meme nombre d'images
                break
            #img_path = classPath + '\\' + img
            img_path = os.path.join(classPath,img) #get image path
            print(img_path)
            resized_img = resize_img(img_path,size) #to resize our new image
            
            class_dir_name = 'class' + str(class_count)    #create new class dir 
            class_out_dir = os.path.join(out_dir,class_dir_name) #get new class path
            
            if not os.path.exists(class_out_dir):
               os.makedirs(class_out_dir)

            img_name = str(class_count) + 'img_' + str(image_count) + '.jpg' #to change extension and define a label for our images 
            img_out = os.path.join(class_out_dir,img_name) #get image path
      
            cv2.imwrite(img_out,resized_img) #write a new image in new dir
            image_count+=1 
            
        image_count = 0
        
        
    class_count+=1

print("Done !")
        

        
