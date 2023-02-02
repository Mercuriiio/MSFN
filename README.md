# MSFN

Self-supervised Learning-based Multi-scale Feature Fusion Network for Survival Analysis from Whole Slide Images

## Data
The training and testing data can be found in the supplementary material, which you can download them at: [data](https://portal.gdc.cancer.gov/)

## Models
The pre-tained models of MSFN are available at: [models](https://drive.google.com/drive/folders/1O1htcCJMfE5UXwI4w_9WjMqzeDrrpgba?usp=sharing)

## Data preprocessing
### Obtain the slide
```
python ./pre_processing/slide.py
```
### Obtain the patches
```
python ./pre_processing/patch.py
```

## Test
```
python test.py
```
