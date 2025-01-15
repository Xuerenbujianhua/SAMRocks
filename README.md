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