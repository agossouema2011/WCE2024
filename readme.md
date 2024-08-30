# Influence of color correction on pathology detection in Capsule Endoscopy
This is an official implementation for WCE Workshop in ICPR paper ["Influence of color correction on pathology detection in Capsule Endoscopy"](https://xx.pdf). 
It includes the code for Object Detection tasks using Retinanet and YOLOv5 on the SEE-AI Capsule Endoscopy dataset.
(1) Retinant and YOLOv5 benchmarked on the original SEE-AI Dataset (R_OrigD)
(2) Retinant and YOLOv5 benchmarked on the Color Checker Corrected version of SEE-AI Dataset (R_CCD)
(3) Retinant and YOLOv5 benchmarked on the Colon Checker Corrected version of the SEE-AI Dataset (R_CCCD).
The performance of the two SOA models Retinanet and YOLOv5 on the three datasets is reported in this repo.


## Dataset Preparation

### For Retinanet Model:
```
dataset/
  csv/
    class_list.csv
    train_annots.csv
    val_annots.csv
  images/
    training/
    validation/
```
You can download the corresponding dataset for each of the three experiment from the following link:
https://www.kaggle.com/datasets/emmanuelagosou/see-ai-dataset-r-origd-retinanet
https://www.kaggle.com/datasets/emmanuelagosou/see-ai-dataset-r-ccd-retinanet
https://www.kaggle.com/datasets/emmanuelagosou/see-ai-dataset-r-cccd-retinanet

You will have to unzip and copy and paste the content in the corresponding folder.


### For YOLOV5 Model:
```
data/
  images/
    training/
    validation/
  labels/
    training/
    validation/
   
```
You can download the corresponding dataset for each of the three experiment from the following link:
https://www.kaggle.com/datasets/emmanuelagosou/see-ai-datatset-r-origd-yolo
https://www.kaggle.com/datasets/emmanuelagosou/see-ai-datatset-r-ccd-yolo
https://www.kaggle.com/datasets/emmanuelagosou/see-ai-datatset-r-cccd-yolo

You will have to unzip and copy and paste the content in the corresponding folder.


## Training Retinanet Model

```
python train.py --dataset csv --csv_train dataset/csv/train_annots.csv  --csv_classes dataset/csv/class_list.csv --csv_val dataset/csv/val_annots.csv --depth 50 --epochs 400

```
## Validation for Retinanet Model

```
python csv_validation.py --csv_annotations_path dataset/csv/val_annots.csv  --model_path outputs/csv_retinanet_182.pt   --class_list_path dataset/csv/class_list.csv

```

## Testing Retinanet Model

```
python visualize_single_image_advanced.py --csv_annotations_path dataset/csv/val_annots.csv  --images_dir images/myimages --class_list dataset/csv/class_list.csv --model_path outputs/csv_retinanet_182.pt  --depth 50  
```
## Training YOLOv5 Model

```
python yolov5/train.py --img 576 --batch 32 --epochs 400 --data dataset.yaml --weights yolov5s.pt 
```
## Validation for YOLOv5 Model

```
python yolov5/val.py --weights yolov5/runs/train/exp3/weights/best.pt --data dataset.yaml --img 576

```
## Testing YOLOv5 Model

```

python yolov5/detect.py --weights yolov5/runs/train/exp3/weights/best.pt --conf_thres 0.24 --iou_thres 0.5 --source data/images/validation/  --annotations_path data/labels/validation
```
## Performance metrics

 Table 1. Models performances in F1-scores across the three different color schemes. Bold values indicate wherever color-correction results in better 

|                  |         | Retinanet |       |        | YOLOv5  |       |        |
|------------------|---------|-----------|-------|--------|---------|-------|--------|
| Pathologies      | Weights | R-OrigD   | R-CCD | R-CCCD | R-OrigD | R-CCD | R-CCCD |
| angiodysplasia   | 0.036   | 0.444     | 0.523 | 0.515  | 0.629   | 0.601 | 0.634  |
| erosion          | 0.252   | 0.623     | 0.567 | 0.581  | 0.676   | 0.667 | 0.677  |
| stenosis         | 0.021   | 0.564     | 0.549 | 0.515  | 0.691   | 0.747 | 0.657  |
| lymphangiectasia | 0.025   | 0.587     | 0.746 | 0.662  | 0.718   | 0.776 | 0.759  |
| lymph-follicle   | 0.299   | 0.538     | 0.535 | 0.539  | 0.584   | 0.583 | 0.563  |
| SMT              | 0.024   | 0.522     | 0.582 | 0.449  | 0.736   | 0.705 | 0.733  |
| polyp-like       | 0.144   | 0.57      | 0.572 | 0.593  | 0.697   | 0.718 | 0.696  |
| bleeding         | 0.049   | 0.643     | 0.552 | 0.589  | 0.693   | 0.65  | 0.677  |
| diverticulum     | 0.003   | 0.133     | 0.211 | 0.273  | 0.301   | 0.159 | 0.175  |
| erythema         | 0.048   | 0.435     | 0.36  | 0.397  | 0.492   | 0.403 | 0.476  |
| foreign-body     | 0.068   | 0.593     | 0.545 | 0.537  | 0.697   | 0.643 | 0.68   |
| vein             | 0.031   | 0.739     | 0.747 | 0.773  | 0.816   | 0.797 | 0.761  |
| F1-score         |         | 0.571     | 0.553 | 0.559  | 0.649   | 0.638 | 0.638  |


Table 2. Models performances in AP across the three different experiments schemes. Bold values indicate wherever color-correction results in better scores.

|                  |         | RetinaNet |       |        | YOLOv5  |       |        |
|------------------|---------|-----------|-------|--------|---------|-------|--------|
| Pathologies      | Weights | R-OrigD   | R-CCD | R-CCCD | R-OrigD | R-CCD | R-CCCD |
| angiodysplasia   | 0.036   | 0.468     | 0.486 | 0.485  | 0.627   | 0.595 | 0.608  |
| erosion          | 0.252   | 0.572     | 0.58  | 0.577  | 0.695   | 0.684 | 0.706  |
| stenosis         | 0.021   | 0.895     | 0.834 | 0.828  | 0.766   | 0.827 | 0.765  |
| lymphangiectasia | 0.025   | 0.676     | 0.672 | 0.649  | 0.761   | 0.79  | 0.814  |
| lymph-follicle   | 0.299   | 0.469     | 0.463 | 0.449  | 0.622   | 0.588 | 0.597  |
| SMT              | 0.024   | 0.611     | 0.703 | 0.743  | 0.791   | 0.785 | 0.765  |
| polyp-like       | 0.144   | 0.516     | 0.538 | 0.535  | 0.705   | 0.756 | 0.74   |
| bleeding         | 0.049   | 0.643     | 0.643 | 0.679  | 0.706   | 0.709 | 0.714  |
| diverticulum     | 0.003   | 0.222     | 0.133 | 0.278  | 0.19    | 0.213 | 0.173  |
| erythema         | 0.048   | 0.496     | 0.543 | 0.504  | 0.536   | 0.464 | 0.497  |
| foreign-body     | 0.068   | 0.496     | 0.525 | 0.541  | 0.727   | 0.706 | 0.739  |
| vein             | 0.031   | 0.804     | 0.778 | 0.882  | 0.889   | 0.849 | 0.809  |
| AP               |         | 0.54      | 0.548 | 0.548  | 0.677   | 0.666 | 0.674  |

Table 3. Retinanet: IoU comparison across the three different color schemes. Green arrows indicate wherever there is an increase of the metric value whereas red arrows indicate wherever there is a decrease.
|                                                |         |         |         |       |      |
|------------------------------------------------|---------|---------|---------|-------|------|
| Pathologies                                    | R-OrigD | R-CCD   | R-CCCD  | % I1  | % I2 |
| angiodysplasia                                 | 0.284   | 0.362 ↑ | 0.342 ↑ | 7.8   | 5.8  |
| erosion                                        | 0.524   | 0.434 ↓ | 0.448 ↓ | -9    | -7.6 |
| stenosis                                       | 0.367   | 0.338 ↓ | 0.325 ↓ | -2.9  | -4.2 |
| lymphangiectasia                               | 0.391   | 0.624 ↑ | 0.487 ↑ | 23.3  | 9.6  |
| lymph-follicle                                 | 0.339   | 0.342 ↑ | 0.383 ↑ | 0.3   | 4.4  |
| SMT                                            | 0.342   | 0.383 ↑ | 0.260 ↓ | 4.1   | -8.2 |
| polyp-like                                     | 0.452   | 0.462 ↑ | 0.486 ↑ | 1     | 3.4  |
| bleeding                                       | 0.536   | 0.429 ↓ | 0.442 ↓ | -10.7 | -9.4 |
| diverticulum                                   | 0.116   | 0.165 ↑ | 0.200 ↑ | 4.9   | 8.4  |
| erythema                                       | 0.288   | 0.218 ↓ | 0.246 ↓ | -7    | -4.2 |
| foreign-body                                   | 0.468   | 0.365 ↓ | 0.396 ↓ | -10.3 | -7.2 |
| vein                                           | 0.585   | 0.607 ↑ | 0.580 ↓ | 2.2   | -0.5 |
| %I1 is the increase with R-CCD against R-OrigD |         |         |         |       |      |
| %I2 is the increase of R-CCCD against R-OrigD  |         |         |         |       |      |
|                                                |         |         |         |       |      |


## False Positive Results 
```

```

## Bounding Boxes Areas Results 
```

```

## Some Detection Results 
```

```


<table border="0" width="800">
<tr>
	
</tr>
	
<tr>
	
</tr>
<tr>
	<td width="15%" align="center"> 2% </td>
	<td width="25%" align="center"> 24.99 </td>
</tr>
<tr>
	<td width="15%" align="center"> 5% </td>
	<td width="25%" align="center"> 30.07 </td>
</tr>
<tr>
	<td width="15%" align="center"> 10% </td>
	<td width="25%" align="center"> 32.58 </td>
</tr>
<tr>
	<td width="15%" align="center"> 20% </td>
	<td width="25%" align="center"> 35.49 </td>
</tr>

</table>
	
## Citing the paper

If you find this work useful in your research, please consider citing:

```
@InProceedings{,
}   
```

## License

This work is released under the [xxxx](LICENSE).

## Acknowledgement
-   [Retinanet](https://github.com/yhenon/pytorch-retinanet)
-   [YOLOv5](https://github.com/ultralytics/yolov5/tree/master)
- Enhancement of Colour Reproduction for Capsule Endoscopy Images
```
@inproceedings{watine2023enhancement,
  title={Enhancement of Colour Reproduction for Capsule Endoscopy Images},
  author={Watine, L{\'e}o and Floor, P{\aa}l Anders and Pedersen, Marius and Nussbaum, Peter and Ahmad, Bilal and Hovde, {\O}istein},
  booktitle={2023 11th European Workshop on Visual Information Processing (EUVIP)},
  pages={1--6},
  year={2023},
  organization={IEEE}
```
}

