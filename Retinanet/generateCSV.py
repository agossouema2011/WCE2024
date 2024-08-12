import os
import glob
import pandas as pd
import numpy as np


#os.chdir(r'dataset\images\training') # Put the directory location here: training or validation
os.chdir(r'dataset\images\validation')
myFiles = glob.glob('*.txt')
class_name=['angiodysplasia','erosion','stenosis','lymphangiectasia','lymph-follicle','SMT','polyp-like',
               'bleeding','diverticulum','erythema','foreign-body','vein'] #Pathologies names list

final_df=[] # array for the CSV file
# for each text file in the directory
for item in myFiles:
    # open each text file to read the content and build the csv file content
    with open(item, 'rt') as fd:
        for line in fd.readlines(): #read each line of the txt file
           theline=""
           filename = os.path.basename(item).split(".")[0] # get the file mane without the extension
           location="dataset/Bleeding/training/"+filename+".jpg" # the corresponding image file location
           #print("File location:", location)
           theline=theline+location
           splited = line.split()
           try: #get each element of the line
               theline=theline+","+splited[1]
               theline = theline + "," + splited[2]
               theline = theline + "," + splited[3]
               theline = theline + "," + splited[4]
               theline = theline + "," + class_name[int(splited[0])]
               final_df.append(theline)
           except:
               print("file is not in YOLO format!")

df = pd.DataFrame(final_df)
#df.to_csv("../../csv/train_annots.csv",index=False)
df.to_csv("../../csv/val_annots.csv",index=False)