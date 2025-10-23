# SAMRocks
《Few-Shot Intelligent Identification of Rock Thin Sections Based on SAM》
SAMRocks is a strategy that utilizes the SAM model in combination with an image classification model to accomplish semantic segmentation on rock thin-section images in low-data scenarios. It aims to address the shortage of labeled data and the difficulty of mineral grain segmentation in rock thin-section image analysis.

# Installation
1. Follow the [SAM official documentation](https://github.com/facebookresearch/segment-anything) to set up the environment.
2. Then install the required libraries for this environment:

```
pip install -r requirements.txt
```

# File Structure and Description
- SAMRocks
  - checkpoints (Stores model checkpoint files)
  - data (Stores project data)
    - inputdata (Contains images and corresponding JSON label files)
    - testdata (Contains test set rock thin-section images)
    - traindata
      - images (Original rock thin-section images)
      - labels (Image label files)
      - sams (Images segmented by the SAM model)
  - Dataset (Data processing module)
  - Main
    - model_result (Stores model predictions and evaluation results)
    - results (Stores outputs for grain extraction, etc.)
    - One_Data_preprocessing.ipynb
    - Two_Train_models.ipynb
    - Three_Use_models.ipynb
  - Mineral_segmentation (Module for grain segmentation, stitching, etc.)
  - scrips (Configuration files)
  - segment-anything-main (SAM model directory)


# How to Run
1. Prepare your rock thin-section images and their corresponding JSON files. Make sure no Chinese characters are used in paths or filenames. Place them into the `inputdata` folder.
2. In the `scrips` folder, configure the mineral classes and their color mappings.
3. Run `One_Data_preprocessing.ipynb` to convert JSON files into labeled images and generate SAM masks.
4. Run `Two_Train_models.ipynb` to start training the model.
5. Run `Three_Use_models.ipynb` to use the trained model to predict on new rock thin-section images.

# Data  
The datasets used in this project are derived from two publicly available datasets, aiming to validate the algorithm's usability and transferability while ensuring the reliability and reproducibility of the experimental results. These datasets are sourced from [Micro image data set of some rock forming minerals, typical metamorphic minerals and oolitic thin sections](https://www.scidb.cn/en/detail?dataSetId=684362351280914432&language=zh_CN&dataSetType=journal) and [A photomicrograph dataset of rocks for petrology teaching in Nanjing University](http://www.csdata.org/p/474/2/). Their public availability ensures data transparency and provides a standardized reference for subsequent research. By utilizing publicly available datasets, the project better evaluates the algorithm's performance across different scenarios and data types, providing guidance for related research and applications.  

Data Availability Statement
This dataset is licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0).
You are free to use, modify, and redistribute the data provided that:
1.you give appropriate credit to the original author(s);
2.your use is non-commercial;
3.any derivative works are shared under the same license.
The dataset is available at:https://www.kaggle.com/datasets/xuerenjianhua/rock-slice-dataset-and-labels

# Model Training Hyperparameter Settings

|  Parameter Name| img size |batch size|Learning rate | Area |Line Threshold | Data Augmentation Factor|
|--|--|--|--|--|--|--|
| Parameter Setting |224*224  |32 |0.01|150	|0.15	|5

- img size：Image processing size. Exceeding this size may cause memory overflow
 - Area ： Minimum area filtering (pixel area), segmentation results smaller than this area will be filtered, eliminating data redundancy while ensuring the reliability of SAM segmentation results
 - Line Threshold ：The line filtering threshold ranges from 0 to 1. Image segmentation results smaller than the set value will be filtered, removes narrow edges without feature information to prevent data redundancy and improve model performance
 - Data Augmentation Factor, the multiple of data enhancement. Exceeding this multiple will lead to an increase in training costs, and when the enhancement multiple is too large, the performance improvement is not significant

# License
Copyright 2025  Zhuofeng Zhang

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

     http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
