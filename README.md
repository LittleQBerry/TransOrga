# TransOrga
 
### Organoid research plays an important role in drug screening and disease modeling. Obtaining accurate information about organoid morphology, number, and size is fundamental to this research. However, previous methods relied on fluorescence labeling which can harm organoids or have problems with accuracy and robustness. In this paper, we first introduce Transformer architecture into the organoid segmentation task and propose an end-to-end multi-modal method named TransOrga. To enhance the accuracy and robustness, we utilize a multi-modal feature extraction module to blend spatial and frequency domain features of organoid images. Furthermore, we propose a multi-branch aggregation decoder that learns diverse contexts from various Transformer layers to predict the segmentation mask progressively. In addition, we design a series of losses, including focal loss, dice loss, compact loss and auxiliary loss, to supervise our model to predict more accurate segmentation results with rational sizes and shapes. Our extensive experiments demonstrate that our method outperforms the baselines in organoid segmentation and provides an automatic, robust, and fluorescent-free tool for organoid research.

![Network](https://github.com/LittleQBerry/TransOrga/blob/main/image/network.png)

# Environment

pytorch 1.10.0 
scikit-image 0.20.0
numpy 1.23.5
opencv-python 4.7.0.72
scikit-learn 1.2.2
monai 1.1.0
d2l 0.17.6
timm 0.6.12
einops 0.6.0

# Dataset
The dataset for model training and validation/testing is openly available here: https://osf.io/xmes4/
Please download the dataset and put them under /dataset
# Preprocess

The frequency-level images of https://osf.io/xmes4/ is [SR_result](https://drive.google.com/file/d/1F0eUE39K6k09U5Ib7aHPvmsgOzLD-U5_/view?usp=sharing) and the edge images is [edge](https://drive.google.com/file/d/1DJslK0MAXTmoflxycCBbMOZKXjpCqMJc/view?usp=sharing).

Please unzip these files and place the folders in the root directory, like 

'TransOrage/Dataset',

'TransOrga/SR_results', 

'TransOrga/edge_results'.

If you want to utilize your data: Please first obtain the frequency-level images from **[SRNET.py](SRNet.py)**

# Test

The pretrained model is [here](https://drive.google.com/file/d/1c6Ka99uWFOBYwN325Q9FjARW9d-iAeNQ/view?usp=sharing) .

1. Download the pretrained model and put it in 'checkpoints/'.

2. create 'log_results/' as the output folder.

3. run **[test.py](test.py)** to obtain the results.



# Train
1. **[dataset.py](dataset.py)** contains how to load the data.
2. run **[train.py](train.py)**



# Result
![Compare](https://github.com/LittleQBerry/TransOrga/blob/main/image/compare_new.png)

![MoreResult](https://github.com/LittleQBerry/TransOrga/blob/main/image/cased.png)
ACC organoid
| Model | Dice ↑ | mIoU↑|  Precision↑ |Recall↑ | F1-score↑ |
| :----: | :----: | :----: |:----: |:----: |:----: |
| SegNet | 0.798 | 0.664 |0.579 |0.803 |0.630 |
| A-Unet | 0.884 | 0.791 |0.671 |0.952 |0.783 |
| OrganoID | 0.848 | 0.736 |0.622 |0.866|0.716|
| Ours | 0.913 | 0.840 |0.791|0.903|0.837 |

Colon organoid
|   Model  | Dice↑ | mIoU↑ | Precision↑ | Recall↑ | F1-score↑ |
|:--------:|:-----:|:-----:|:----------:|:-------:|:---------:|
|  SegNet  | 0.864 | 0.761 |    0.742   |  0.769  |   0.738   |
|  A-Unet  | 0.877 | 0.781 |    0.645   |  0.952  |   0.764   |
| OrganoID | 0.867 | 0.766 |    0.674   |  0.850  |   0.745   |
|   Ours   | 0.919 | 0.851 |    0.786   |  0.918  |   0.844   |

Lung organoid
|     Model    | Dice↑ | mIoU↑ | Precision↑ | Recall↑ | F1-score↑ |
|:------------:|:-----:|:-----:|:----------:|:-------:|:---------:|
|  SegNet  | 0.877 | 0.781 |    0.903   |  0.730  |   0.801   |
|  A-Unet  | 0.948 | 0.900 |    0.892   |  0.946  |   0.917   |
| OrganoID | 0.911 | 0.836 |    0.794   |  0.938  |   0.858   |
|     Ours     | 0.946 | 0.898 |    0.921   |  0.910  |   0.915   |

PDAC organoid
|   Model  | Dice↑ | mIoU↑ | Precision↑ | Recall↑ | F1-score↑ |
|:--------:|:-----:|:-----:|:----------:|:-------:|:---------:|
|  SegNet  | 0.875 | 0.778 |    0.740   |  0.855  |   0.783   |
|  A-Unet  | 0.889 | 0.801 |    0.763   |  0.875  |   0.806   |
| OrganoID | 0.859 | 0.752 |    0.702   |  0.836  |   0.752   |
|   Ours   | 0.898 | 0.814 |    0.778   |  0.885  |   0.821   |

