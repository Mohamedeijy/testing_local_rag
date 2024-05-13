---
References: "[[Modern Computer Vision with PyTorch]]"
Subjects: "[[Deep Learning]]"
tags:
  - computervision
  - neuralnetwork
  - objectdetection
  - rcnn
  - fast_rcnn
  - faster_rcnn
---
R-CNN stands for **Region-based Convolutional Neural Network** ([original paper]([arxiv.org/pdf/1311.2524.pdf](https://arxiv.org/pdf/1311.2524.pdf))). Region-based within R-CNN stands for the region proposals. Region proposals are used to identify objects within an image. Note that R-CNN assists in identifying both the objects present in the image and the location of objects within the image.

The high level workings of an R-CNN are as follows:
![[Pasted image 20231005164744.png]]

We perform the following steps when leveraging the R-CNN technique for object detection: 

1. Extract region proposals from an image: Ensure that we extract a high number of proposals to not miss out on any potential object within the image. 

2. Resize (warp) all the extracted regions to get images of the same size. 

3. Pass the resized region proposals through a network: Typically, we pass the resized region proposals through a pretrained model such as VGG16 or ResNet50 and extract the features in a fully connected layer. 

4. Create data for model training, where the input is features extracted by passing the region proposals through a pretrained model, and the outputs are the class corresponding to each region proposal and the offset of the region proposal from the ground truth corresponding to the image: If a region proposal has an IoU greater than a certain threshold with the object, we prepare training data in such a way that the region is responsible for predicting the class of object it is overlapping with and also the offset of region proposal with the ground truth bounding box that contains the object of interest. A sample as a result of creating a bounding box offset and a ground truth class for a region proposal is as follows:        ![[Pasted image 20231005165429.png]] 
   In the preceding image, o (in red) represents the center of the region proposal (dotted bounding box) and x represents the center of the ground truth bounding box (solid bounding box) corresponding to the cat class. We calculate the offset between the region proposal bounding box and the ground truth bounding box as the difference between center co-ordinates of the two bounding boxes (dx, dy) and the difference between the height and width of the bounding boxes (dw, dh). 

5. Connect two output heads, one corresponding to the class of image and the other corresponding to the offset of region proposal with the ground truth bounding box to extract the fine bounding box on the object: This exercise would be similar to the use case where we predicted gender (categorical variable, analogous to the class of object in this case study) and age (continuous variable, analogous to the offsets to be done on top of region proposals) based on the image of the face of a person in the previous chapter. 

6. Train the model post, writing a custom loss function that minimizes both object classification error and the bounding box offset error.


We will go over the different steps for implementing R-CNN for object detection with a custom dataset.

## Implementing R-CNN with a custom dataset

The steps for creating a custom dataset for train an R-CNN are as follows:
- Prepare the dataset object (to allow for fetching of the image, its classes and the bounding boxes)
- Define the region proposals extraction function and the IoU calculation function
- Create the training data (resizing the region proposals for inputs, label each region proposal with a class or background label and deffine the offest between region and ground truth for the output)
- Train our model
- Evaluate

## R-CNN network architecture

The dataset provides not only the cropped and resized region proposal generated through the segmentation process (to predict its class), but also the delta/offset between the region proposal and the ground truth (to predict a tight bounding box). Our R-CNN network has VGG backbone, and multivariate prediction (just like the previous example in the book, for gender and age estimation).

![[Pasted image 20231009183307.png]]

This prediction process has good results, but takes a lot of time. Most of that time is spent generating countless region proposals, resizing them and passing each one of them trough the VGG backbone to extract features. 

## Fast R-CNN based custom objet detectors

Fast R-CNN gets around this slowness problem by passing the entire image through our CNN based model and then generating the region proposals over the extracted features (which are still obtained through `selectivesearch`, which is still a costly method). Those region proposals (regions of interest) off the extracted features are then put through a classifier, after a manipulation called **RoI Pooling**, a pooling method that makes every region have the same size, in replacement to the warping/resize used in the classical R-CNN approach.

![[Pasted image 20231009183623.png]]

![[Pasted image 20231009183742.png]]

The bottleneck still exists here, as our architecture is split into the region proposal model with the CNN and the `selectivesearch` algorithm, and the object detection model. This is where the Faster R-CNN approach comes into play. 

## Faster R-CNN working details can be found [[Advanced Object Detection#Components of modern object detection algorithms/ Faster R-CNN architecture|here]]
