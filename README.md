# Material Database Classification Using CNN Architecture

Sure, here's the content for your README file:

```markdown
# Material Database Classification Using CNN Architecture

## Group Level - B
Authors:
- Jhuma Mim
- Yang Liu
- Bipul Biswas

Date: 12.2022

## Contents
1. [Introduction](#introduction)
2. [Feature Extraction](#feature-extraction)
3. [Data Processing](#data-processing)
4. [Models](#models)
5. [Results and Model Evaluation](#results-and-model-evaluation)
6. [Conclusion](#conclusion)

## Introduction
In this work, we used the material image database for studying human material categorization for training and testing material recognition systems. This database was constructed with the specific purpose of capturing a range of real-world appearances of everyday materials (e.g., glass, plastic, etc.). Each image in this database (100 pictures per category, ten categories) was selected manually from Flickr.com to ensure various illumination conditions, compositions, colors, texture, and material sub-types.

### Problem Statement
Compared to human performance in the original Flickr Material Database (FMD) pictures, the performance of present vision systems falls short. The proposed approach is developed, and the accuracy attained is comparable to that of humans on databases such as FMD, allowing the materials to be recognized.

### Objectives
The objective is to be able to classify the data from 10 classes, namely Fabric, Foliage, Glass, Leather, Metal, Paper, Plastic, Stone, Water, and Wood, by using Convolution Neural Networks (CNN) as a baseline and finally compared with a more advanced approach like CNN with pre-trained model GoogleNet and ResNet-50. Different tasks will be done, starting from data pre-processing to the classification model to achieve the goal. To improve the accuracy of recognition of materials over Flicker Materials Database. The tasks will be processed in the way they follow the requirements of the approach.

## Feature Extraction
It is the process of taking raw data and extracting valuable features for modeling. With images, this usually means extracting things like color, texture, shape, etc. We will do a different kind of dataset processing to improve the features as below. Finally, during training, we think that we will use extracting features such as GoogleNet(Inception-V3), VGGNet-19 ResNet-50 networks; these networks were pre-trained on the enormous data set known as the ImageNet dataset used in image classification tasks.

## Data Processing
Before starting the experiment, we read the dataset and have provided labels for each category from(0 to 9).

### Our dataset Balance or Imbalance?
We notice our dataset is a balanced class which means each class has the same number of images. We have 10 classes, and each class has 100 images.

### Do we need data augmentation?
Though our dataset is balanced. We see that we have only 1000 samples; therefore, we experimented with augmenting the dataset. After data augmentation, our best model performance improved by 0.5%. There are several works that mention model performance improvement after data augmentation as well. Some data augmentation we have placed are examples, Reshaping images to lower dimensions, Flipping (vertical, horizontal), Rotation, and Grayscale conversion.

#### Dimension reduction
This is one of the image augmentation approaches. The image will be reduced in size and will help memory issues and faster runtime. Since our images are more than 500 px in size, therefore, we think a reduction of size will help memory issues and faster runtime. We experimented with different dimension reductions and found several image classifications done even with the smallest dimension 28 * 28 dimensions. However, since imageNet required 224 * 224 and reducing too much can be a reason to lose essential features, therefore throughout our experiment, we have used this dimension.

## Models
We used a total of three models in our experiments (i) Basic CNN, (ii) CNN with pre-trained imageNet (iii) CNN with pre-trained ResNet50. In all of our experiments, we have used the same preprocessing and data augmentation techniques so that we can compare final results.

### Basic CNN
We have created a basic CNN model that has an Image Input Layer →Convolution 2D Layer →ReLU Layer →Max Pooling Layer with 2D convolution Layer →Fully Connected Layer with 10 Classes →Softmax Layer and →Classification Layer The final layer. The image input layers take 224 * 224 dimension RGB image that we have used for imageNET and ResNe50. We used different filter sizes 8, 16, and 32, during the experiment. We have used shuffle in every epoch and validation iteration 3. We have tested with different layers and epochs (10 to 100) and optimizers (adam and sgdm). The best result of parameter tuning has mentioned in the results section.

### CNN with pre-trained imageNet
GoogLeNet is type of convolutional neural network based on the Inception architecture. It makes use of Inception modules, which provide the network the ability to select from a variety of convolutional filter sizes in each block. An Inception network layers these modules on top of each other, with max-pooling layers with stride 2 on occasion to reduce the grid’s resolution. The GoogleNet Architecture is 22 layers deep, with 27 pooling layers included. There are 9 inception modules stacked linearly in total. The ends of the inception modules are connected to the global average pooling layer.

### CNN with pre-trained ResNet50
In this architecture, we have used pre-trained Resnet50 the same way as imageNet with CNN architecture. We have replaced the last three layers for transfer retraining and Connected the last transfer layer to new layers.

## Results and Model Evaluation
The dataset has 1000 pictures of 10 categories in total. We divide the training set and validation data set according to the ratio of 8:2. From the results, we see that the basic CNN model has performed lowest compared to the pre-trained model CNN with imageNet and resNet50. It has only 40% train accuracy and 38.5% validation accuracy. On the other hand, both imageNet and ResNet50 have performed similarly with 78.5% and 79% validation accuracy. Though we see a little 0.5% outperformance for resNet50 compared to imageNet, however, we observed little overfitting issues for the resNet50 model compared to imageNet.

## Conclusion
In this project work we classify the material data of 10 classes by using Convolution Neural Networks (CNN) as a baseline and finally compare with a more advanced approach like CNN with pre-trained model GoogleNet and ResNet50. We did different preprocessing, and among flipping and rotation did improve classifier performance; however, image grayscale conversion did not help. Among the model selection, we found CNN without any pre-trained performed worse; however, imageNet and resNet50 performed very well with 79% of accuracy. We did observe little overfitting issues, and especially two classes (paper and plastic) were confused with each other. After the classification result, model learning, and confusion matrix analysis, we can strongly say that our model is performing well despite having a small dataset. Since we have 10 classes and only have 1000 samples total for training and validation, that means we had only 80 samples for each class to train the model. We believe if we have more data samples, then the minor overfitting issues and less performance of paper and plastic classes could be solved.

## References
[1] L. Sharan, R. Rosenholtz, and E. H. Adelson, “Material perception: What can you see in a brief glance?,” Journal of Vision, vol. 9, pp. 784–784, 2010.
[2] A. Fawzi, H. Samulowitz, D. Turaga, and P. Frossard, “Adaptive data augmentation for image classification,” in 2016 IEEE international conference on image processing (ICIP), pp. 3688–3692, Ieee, 2016.
[3] H. Xiao, K. Rasul, and R. Vollgraf, “Fashion-mnist: a novel image dataset for benchmarking machine learning algorithms,” arXiv preprint arXiv:1708.07747, 2017.
[4] C. Szegedy, W. Liu, Y. Jia, P. Sermanet, S. Reed, D. Anguelov, D. Erhan, V. Vanhoucke, and A. Rabinovich, “Going deeper with convolutions,” in Proceedings of the IEEE conference on computer vision and pattern recognition, pp. 1–9, 2015.
```
 
