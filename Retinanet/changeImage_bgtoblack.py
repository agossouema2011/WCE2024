import cv2
import numpy as np
import os

# Importing Image and ImageChops module from PIL package
from PIL import Image, ImageChops
# specify the img directory path
path = "dataset-old/images/training"

# list files in img directory
files = os.listdir(path)

for file in files:
    # make sure file is an image
    if file.endswith(('.jpg', '.png', 'jpeg', 'jpg')):
        thefilepath=path+"/"+file
        image = Image.open(thefilepath)
        # background image
        bg = Image.open(r"theBG.jpeg")


        # applying multiply method
        im3 = ImageChops.multiply(bg, image)
        file_name = thefilepath.split("/")[-1]
        print("File name: ",file_name)
        #im3.show()
        #cv2.imwrite('new/validation/'+file_name, im3)  # Save the output image.
        im3 = im3.save('dataset/images/training/'+file_name)