import torch
import numpy as np

import time
import os
import csv
import cv2
import argparse
import matplotlib.pyplot as plt
from retinanet import model
# from retinanet.dataloader import CSVDataset, Resizer, Normalizer
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from retinanet.dataloader import CocoDataset, CSVDataset, collater, Resizer, AspectRatioBasedSampler, Augmenter, \
    UnNormalizer, Normalizer
from skimage import io


def load_classes(csv_reader):
    result = {}

    for line, row in enumerate(csv_reader):
        line += 1

        try:
            class_name, class_id = row
        except ValueError:
            raise (ValueError('line {}: format should be \'class_name,class_id\''.format(line)))
        class_id = int(class_id)

        if class_name in result:
            raise ValueError('line {}: duplicate class name: \'{}\''.format(line, class_name))
        result[class_name] = class_id
    return result


# Draws a caption above the box in an image
def draw_caption(image, box, caption):
    b = np.array(box).astype(int)
    cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)
    cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)


def draw_caption_GT(image, box, caption):
    b = np.array(box).astype(int)
    cv2.putText(image, caption, (b[2], b[3] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)
    cv2.putText(image, caption, (b[2], b[3] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)


def detect_image(retinanet, image_path, csv_annotations_path, class_list):
    with open(class_list, 'r') as f:
        classes = load_classes(csv.reader(f, delimiter=','))

    labels = {}
    for key, value in classes.items():
        labels[value] = key

    if torch.cuda.is_available():
        retinanet = retinanet.cuda()
        
    #retinanet.training = False
    retinanet.eval()

    # Read the csv file to get the ground truth
    tab_grth = []
    with open(csv_annotations_path, newline='') as f:
        reader = csv.reader(f)
        for row in reader:
            tab_grth.append(row)

    unnormalize = UnNormalizer()

    for img_name in os.listdir(image_path):
        grth_coordinates = []
        # get gound truth coordinates.
        for i in range(len(tab_grth)):
            tab_a_coordinate = []
            # print("tab_grth[i]=",tab_grth[i])
            # print("img_name=", img_name)
            if any(img_name in str for str in tab_grth[i]):
                x1 = float(tab_grth[i][1])
                y1 = float(tab_grth[i][2])
                x2 = float(tab_grth[i][3])
                y2 = float(tab_grth[i][4])
                label = tab_grth[i][5]
                tab_a_coordinate.append(x1)
                tab_a_coordinate.append(y1)
                tab_a_coordinate.append(x2)
                tab_a_coordinate.append(y2)
                tab_a_coordinate.append(label)
                grth_coordinates.append(tab_a_coordinate)
        # print("img_name=", img_name, "\n Ground truths=", grth_coordinates)

        #image = cv2.imread(os.path.join(image_path, img_name))
        image = io.imread(os.path.join(image_path, img_name))
        if image is None:
            continue
    
        # Normalize the image
        image = image / 255
        image = image - [0.485, 0.456, 0.406]
        image = image / [0.229, 0.224, 0.225]
        
        """
        # Resize the image -- Proceed with the model detections coordinates
        rows, cols, cns = image.shape
        smallest_side = min(rows, cols)
        # rescale the image so the smallest side is min_side
        min_side = 608
        max_side = 1024
        scale = min_side / smallest_side
        
        # check if the largest side is now greater than max_side, which can happen
        # when images have a large aspect ratio
        largest_side = max(rows, cols)

        if largest_side * scale > max_side:
            scale = max_side / largest_side

        # resize the image with the computed scale
        image = cv2.resize(image, (int(round(cols * scale)), int(round((rows * scale)))))
        
        rows, cols, cns = image.shape
        pad_w = 32 - rows % 32
        pad_h = 32 - cols % 32
        new_image = np.zeros((rows + pad_w, cols + pad_h, cns)).astype(np.float32)
        new_image[:rows, :cols, :] = image.astype(np.float32)
        """
        #image = new_image
        image = np.expand_dims(image, 0)
        image = np.transpose(image, (0, 3, 1, 2))

        with torch.no_grad():

            image = torch.from_numpy(image)
            st = time.time()
            if torch.cuda.is_available():
                scores, classification, transformed_anchors = retinanet(image.cuda().float())
            else:
                scores, classification, transformed_anchors = retinanet(image.float())

            idxs = np.where(scores.cpu() >= 0.25)
            img = np.array(255 * unnormalize(image[0, :, :, :]))

            img[img < 0] = 0
            img[img > 255] = 255

            img = np.transpose(img, (1, 2, 0))

            img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB)

            
        for j in range(idxs[0].shape[0]):
            bbox = transformed_anchors[idxs[0][j], :]
            """
            x1 = int(bbox[0] / scale)
            y1 = int(bbox[1] / scale)
            x2 = int(bbox[2] / scale)
            y2 = int(bbox[3] / scale)
            """
            x1 = int(bbox[0] )
            y1 = int(bbox[1])
            x2 = int(bbox[2] )
            y2 = int(bbox[3])
            
            label_name = labels[int(classification[idxs[0][j]])]
            idLabel=int(classification[idxs[0][j]])
            # print(bbox, classification.shape)
            # score = scores[j]
            score = scores[idxs[0][j]]
            caption = '{} {:.3f}'.format(label_name, score)
            # draw_caption(img, (x1, y1, x2, y2), label_name)
            draw_caption(img, (x1, y1, x2, y2), caption)
            cv2.rectangle(img, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=2)
            print("-------- Detection img_name=", img_name, "\n x1=", x1, " y1=", y1, " , x2=", x2, " , y2=", y2,
                  ", label=", label_name, "--------\n")
                  
            # store the detections in a text file in folder labels 
            imageName=img_name.rsplit( ".")[ 0 ] # get the image file base name
            filename="outputs/detectionResultsWithGrth/labels/"+imageName+".txt"
            if not os.path.exists(filename):
                with open(filename, "w") as file:
                    file.write(str(idLabel)+" "+str(x1)+" "+str(y1)+" "+str(x2)+" "+str(y2)+" "+str(score.item()) + "\n")

        # Draw bounding boxes for the ground truths
        for box_num in range(len(grth_coordinates)):
            x1 = int(grth_coordinates[box_num][0])
            y1 = int(grth_coordinates[box_num][1])
            x2 = int(grth_coordinates[box_num][2])
            y2 = int(grth_coordinates[box_num][3])
            label = grth_coordinates[box_num][4]
            draw_caption(img, (x1, y1, x2, y2), label)
            cv2.rectangle(img, (x1, y1), (x2, y2), color=(255, 0, 0), thickness=2)
            print("--------  Ground Truth img_name=", img_name, "\n x1=", x1, " y1=", y1, " , x2=", x2, " , y2=", y2,
                  " Label=", label, "--------\n")

        # cv2.imshow('detections', image_orig)
        # save the detections
        plt.figure(figsize=(20, 30))
        # plt.imshow(img)
        plt.xticks([])
        plt.yticks([])
        # plt.show()

        # print("Directory=",os.path.abspath(os.getcwd()))
        nameFig = os.path.basename(img_name)
        print("Detection completed for:", nameFig)
        outfile = "outputs/detectionResultsWithGrth/" + nameFig
        # print("outfile=",outfile)
        cv2.imwrite(outfile, img)
        # cv2.waitKey(0)
        print('Elapsed time: {}'.format(time.time() - st))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Simple script for visualizing result of training.')

    parser.add_argument('--images_dir', help='Path to directory containing images')
    parser.add_argument('--model_path', help='Path to model')
    parser.add_argument('--class_list', help='Path to CSV file listing class names (see README)')
    parser.add_argument('--csv_annotations_path',
                        help='Path to file containing validation annotations (optional, see readme)')
    parser.add_argument('--depth', help='Resnet depth, must be one of 18, 34, 50, 101, 152', type=int, default=50)
    parser = parser.parse_args()

    dataset_val = CSVDataset(parser.csv_annotations_path, parser.class_list,
                             transform=transforms.Compose([Normalizer(), Resizer()]))
    # sampler_val = AspectRatioBasedSampler(dataset_val, batch_size=1, drop_last=False)
    # dataloader_val = DataLoader(dataset_val, num_workers=1, collate_fn=collater, batch_sampler=sampler_val)

    # Create the model
    if parser.depth == 18:
        retinanet = model.resnet18(num_classes=dataset_val.num_classes(), pretrained=True)
    elif parser.depth == 34:
        retinanet = model.resnet34(num_classes=dataset_val.num_classes(), pretrained=True)
    elif parser.depth == 50:
        retinanet = model.resnet50(num_classes=dataset_val.num_classes(), pretrained=True)
    elif parser.depth == 101:
        retinanet = model.resnet101(num_classes=dataset_val.num_classes(), pretrained=True)
    elif parser.depth == 152:
        retinanet = model.resnet152(num_classes=dataset_val.num_classes(), pretrained=True)
    else:
        raise ValueError('Unsupported model depth, must be one of 18, 34, 50, 101, 152')

    retinanet.load_state_dict(torch.load(parser.model_path)["model_state_dict"])

    detect_image(retinanet, parser.images_dir, parser.csv_annotations_path, parser.class_list)

