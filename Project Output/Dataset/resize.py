import cv2
import os
import sys


#img_path = "rose.jpg"

#img = cv2.imread(img_path)

#width = 200
#height = 200

def resize_img(img_path,size):

    img = cv2.imread(img_path)
    new_img = cv2.resize(img,(size,size))

    return new_img
    

#print("Original size: ",img.shape)

#new_img = cv2.resize(img,(width,height))

#print("New size: ",new_img.shape)

#cv2.imshow("Original",img)

#cv2.imshow("New size",new_img)

#cv2.waitKey(0)
in_dir = str(input("in dir path ?: "))
out_dir = str(input("Out dir name ?: "))
size = str(input("New images size?: "))


if not os.path.exists(out_dir):
    os.makedirs(out_dir)


class_count = 0
image_count = 0

for tsfn in os.listdir(in_dir):

    classPath = in_dir + '/' + tsfn 
    if os.path.isdir(classPath):

        print("Processing class: ", class_count) 
        for img in os.listdir(classPath):

            img_path = classPath + '/' + img
            print(img_path)
            resized_img = resize_img(img_path,size)

            class_out_dir = out_dir + '/class' + str(class_count)    
            if not os.path.exists(class_out_dir):
               os.makedirs(class_out_dir)
            img_out = class_out_dir + '/' + class_count + 'img_' + image_count + '.jpg'
      
            cv2.imwrite(img_out,resized_img)
            image_count+=1
            
        image_count = 0
        
    class_count+=1

print("Done !")
        

        
