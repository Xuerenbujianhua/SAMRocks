# SAMRocks
Few-Shot Intelligent Identification of Rock Thin Sections Based on SAM
SAMRocks is a strategy that utilizes the SAM model in combination with an image classification model to accomplish semantic segmentation on rock thin-section images in low-data scenarios. It aims to address the shortage of labeled data and the difficulty of mineral grain segmentation in rock thin-section image analysis.

# Installation
1. Follow the [SAM official documentation](https://github.com/facebookresearch/segment-anything) to set up the environment.
2. Then install the required libraries for this environment:

```
pip install -r requirements.txt
```

# File Structure and Description
- SAMRocks
  - checkpoints
    - (Stores model checkpoint files)
  - data
    - (Stores project data)
    - inputdata
      - (Contains images and corresponding JSON label files)
    - testdata
      - (Contains test set rock thin-section images)
    - traindata
      - images
        - (Original rock thin-section images)
      - labels
        - (Image label files)
      - sams
        - (Images segmented by the SAM model)
  - Dataset
    - (Data processing module)
  - Main
    - model_result
      - (Stores model predictions and evaluation results)
    - results
      - (Stores outputs for grain extraction, etc.)
    - One_Data_preprocessing.ipynb
    - Two_Train_models.ipynb
    - Three_Use_models.ipynb
  - Mineral_segmentation
    - (Module for grain segmentation, stitching, etc.)
  - scrips
    - (Configuration files)
  - segment-anything-main
    - (SAM model directory)

# How to Run
1. Prepare your rock thin-section images and their corresponding JSON files. Make sure no Chinese characters are used in paths or filenames. Place them into the `inputdata` folder.
2. In the `scrips` folder, configure the mineral classes and their color mappings.
3. Run `One_Data_preprocessing.ipynb` to convert JSON files into labeled images and generate SAM masks.
4. Run `Two_Train_models.ipynb` to start training the model.
5. Run `Three_Use_models.ipynb` to use the trained model to predict on new rock thin-section images.
