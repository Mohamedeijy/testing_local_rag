---
References: "[[Modern Computer Vision with PyTorch]]"
Subjects: "[[Deep Learning]]"
tags:
  - computervision
  - deeplearning
  - objectdetection
  - faster_rcnn
---
# Components of modern object detection algorithms/ Faster R-CNN architecture

In contrast with the R-CNN and Fast R-CNN models, modern object detection approaches focus on training a single neural network and achieve the task in one forward pass for an image (the previous two require as many forward passes as there are region proposal).

## Anchor boxes

Anchor boxes come as a replacement of the region proposal approach through `selectivesearch`.

```
Typically, a majority of objects have a similar shape – for example, in a majority of cases, a bounding box corresponding to an image of a person will have a greater height than width, and a bounding box corresponding to the image of a truck will have a greater width than height. Thus, we will have a decent idea of the height and width of the objects present in an image even before training the model (by inspecting the ground truths of bounding boxes corresponding to objects of various classes). 

Furthermore, in some images, the objects of interest might be scaled – resulting in a much smaller or much greater height and width than average – while still maintaining the aspect ratio (that is, height/width).
```

Once we have a good grasp on the general shape (height, width, height/width ratio) of most bounding boxes, we can generate a collection of bounding boxes we establish as being representative of all the bounding boxes in our dataset. This is usually obtained through K-means clustering of the ground truth bounding boxes present in our images.

How to leverage them: 
1. Slide each anchor box over an image from top left to bottom right. 

2. The anchor box that has a high intersection over union (IoU) with the object will have a label that mentions that it contains an object, and the others will be labeled 0: 

```
Once we obtain the ground truths as defined here, we can build a model that can predict the location of an object and also the offset corresponding to the anchor box to match it with ground truth.
```

![[Pasted image 20231009193356.png]]

In this example, we have two anchor boxes. We slide them through the image and find the locations with the greatest IoU with the ground truths, the other locations being labeled as not containing an object. It is important to note that we have at our disposal many anchor boxes with a varying degree of scaling to account for the size of an object on an image. 

## Region Proposal Network (RPN)

Region Proposal Networks leverage anchor boxes to predict the regions that are likely to contain an object.

```
Imagine a scenario where we have a 224 x 224 x 3 image. Furthermore, let's say that the anchor box is of shape 8 x 8 for this example. If we have a stride of 8 pixels, we are fetching 224/8 = 28 crops of a picture for every row – essentially 28*28 = 576 crops from a picture. We then take each of these crops and pass through a Region Proposal Network model (RPN) that indicates whether the crop contains an image. Essentially, an RPN suggests the likelihood of a crop containing an object.
```

While `selectivesearch` generates region proposal through computations on the pixel values of the image, an RPN generates crops of the image and then determines the likelihood of a crop containing an object. An RPN can be integrated as a part of a greater neural network: `We have a single model to identify regions, identify classes of objects in image, and identify their corresponding bounding box locations.`

#### How does an RPN determine the likelihood of a crop containing an object ?

```
In our training data, we would have the bounding box grount truths correspond to objects. 

We now take each region candidate and compare with the ground truth bounding boxes of objects in an image to identify whether the IoU between a region candidate and a ground truth bounding box is greater than a certain threshold. 
If the IoU is greater than a certain threshold (say, 0.5), the region candidate contains an object, and if the IoU is less than a threshold (say 0.1), the region candidate does not contain an object and all the candidates that have an IoU between the two thresholds (0.1 - 0.5) are ignored while training. 

Once we train a model to predict if the region candidate contains an object, we then perform non-max suppression, as multiple overlapping regions can contain an object.
```

## Next step: classification and regression

We have found a new way to identify the regions of an image that contain objects, and know a way to ensure that the feature maps are the size through RoI Pooling. However, just like in our standard R-CNN approach, their needs to be a way to guaranty the prediction of a tight bounding box around an object, and also simply the class of the object. 

This is where we build on top of our RPN, two modules that each compute a classification (class prediction) task and a regression (bounding box delta prediction) task.

![[Pasted image 20231009200042.png]]

```
Hence, if there are 20 classes in the data, the output of the neural network contains a total of 25 outputs – 21 classes (including the background class) and the 4 offsets to be applied to the height, width, and two center coordinates of the bounding box.
```

The complete architecture is thus as follows:

![[Pasted image 20231009200310.png]]


## Faster R-CNN submodules

The model contains the following key submodules:

![[Pasted image 20231009201156.png]]

- `GeneralizedRCNNTransform` is a simple resize followed by a normalize transformation:
	![[Pasted image 20231009201241.png]]

- `BackboneWithFPN` is a neural network that transforms input into a feature map.

- `RegionProposalNetwork` generates the anchor boxes for the preceding feature map and predicts individual feature maps for classification and regression tasks:
	![[Pasted image 20231009201334.png]]

- `RoIHeads` takes the preceding maps, aligns them using RoI pooling, processes them, and returns classification probabilities for each proposal and the corresponding offsets:
	![[Pasted image 20231009201458.png]]

# [[YOLO (You Only Look Once) approach|Working details of YOLO]]

# [[SSD (Single Shot Detector) approach for object detection|Working details of SSD]]
