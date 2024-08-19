import os, shutil, random

# preparing the folder structure

full_data_path = 'data/obj/'
extension_allowed = '.png'
split_percentage = 80

# Create the directory for all the images
images_path = 'data/images/'
if os.path.exists(images_path):
    shutil.rmtree(images_path)
os.mkdir(images_path)

# Create the directory for all the images annotations
labels_path = 'data/labels/'
if os.path.exists(labels_path):
    shutil.rmtree(labels_path)
os.mkdir(labels_path)

# # Create the directory for the training and validation sets and their annotations files
training_images_path = images_path + 'training/'
validation_images_path = images_path + 'validation/'
training_labels_path = labels_path + 'training/'
validation_labels_path = labels_path + 'validation/'

os.mkdir(training_images_path)
os.mkdir(validation_images_path)
os.mkdir(training_labels_path)
os.mkdir(validation_labels_path)

files = []

ext_len = len(extension_allowed)

# Open the obj folder and select all the images with .jpg extension
for r, d, f in os.walk(full_data_path):
    for file in f:
        if file.endswith(extension_allowed):
            strip = file[0:len(file) - ext_len]
            files.append(strip)

random.shuffle(files)

size = len(files)

split = int(split_percentage * size / 100) # split the data

print("copying training data")
for i in range(split):
    strip = files[i]

    image_file = strip + extension_allowed
    src_image = full_data_path + image_file
    shutil.copy(src_image, training_images_path) # Copy the image into the training set folder for images

    annotation_file = strip + '.txt'
    src_label = full_data_path + annotation_file  #Take it corresponding annotation file
    shutil.copy(src_label, training_labels_path)  # Copy the annotation file in the training dataset folder for annotation

# Copy the remaining images and their annotations into the corresponding folders inside the validation directory
print("copying validation data")
for i in range(split, size):
    strip = files[i]
    image_file = strip + extension_allowed
    src_image = full_data_path + image_file
    shutil.copy(src_image, validation_images_path)

    annotation_file = strip + '.txt'
    src_label = full_data_path + annotation_file
    shutil.copy(src_label, validation_labels_path)

print("finished")