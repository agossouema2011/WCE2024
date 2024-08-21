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

```
## Validation for Retinanet Model

```

```
## Testing Retinanet Model

```

```


## Training YOLOv5 Model

```

```
## Validation for YOLOv5 Model

```

```
## Testing YOLOv5 Model

```

```
## Performance metrics
```

```

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

