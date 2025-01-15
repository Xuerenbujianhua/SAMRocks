# SAMRocks
这是《Few-Shot Intelligent Identification of Rock Thin Sections Based onSAM》的可用代码SAMRocks，
SAMRocks是利用SAM语义分割大模型结合预训练的图像分类模型，完成小样本岩石薄片语义分割任务的策略，
用以解决岩石薄片智能鉴定中标签数据稀缺、矿物颗粒分割困难的问题。


# 安装
首先参照[SAM](https://github.com/facebookresearch/segment-anything)官方文档配置环境,
然后再安装本环境的库
```
pip install -r resuirements.txt
```


# 文件结构及说明
- SAMRocks
  - checkpoints （模型文件储存位置）
  - data （数据存放位置）
      - inputdata （图像及其对应的json标签）
      - testdata  （测试集岩石薄片图像）
      - traindata （训练集岩石薄片图像）
      	- images （原始岩石薄片图像）
      	- labels  （图像标签）
      	- sams  (SAM模型对岩石薄片图像分割后的图像)
  - Dataset （数据处理模块）
  - Main （主函数）
       - model_result （运行结果，储存模型预测及评价结果）
       - results（运行结果，储存颗粒提取等结果）
       - One_Data_preprocessing.ipynb 
       - Two_Train_models.ipynb
       - Three_Use_models.ipynb
   - Mineral_segmentation（颗粒分割拼接等模块）
  - scrips （配置文件）
  - segment-anything-main （SAM模型）



# 运行
（1）准备好岩石薄片图像及其对应的json文件，确保路径及文件名无中文，并放入inputdata 文件夹
（2）在scrips中设置待训练的矿物类别和色标
（3）运行 One_Data_preprocessing.ipynb ，将json文件转化为label图像，并生成sam掩码
（4）运行Two_Train_models.ipynb，开始训练模型
（5）运行Three_Use_models.ipynb，使用训练后的模型对未知的岩石薄片图像进行预测

# 数据
本项目使用的数据集来源于两个公开数据集，旨在验证算法的可用性和可迁移性，以确保实验结果的可靠性与可重复性。这些数据集分别来自 [Micro image data set of some rock forming minerals, typical metamorphic minerals and oolitic thin sections](https://www.scidb.cn/en/detail?dataSetId=684362351280914432&language=zh_CN&dataSetType=journal)和 [A photomicrograph dataset of rocks for petrology teaching in Nanjing University](http://www.csdata.org/p/474/2/)，其公开来源确保了数据的透明性，并为后续研究提供了标准化的参考。通过使用公开数据集，项目能够更好地评估算法在不同场景和数据类型下的性能，为相关领域的研究和应用提供借鉴。

# 许可证
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
