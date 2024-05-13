---
References: "[[Modern Computer Vision with PyTorch]]"
Subjects: "[[Deep Learning]]"
tags:
  - computervision
  - deeplearning
  - objectdetection
  - neuralnetwork
---
```
So far, we have seen a scenario where we made predictions after gradually convolving and pooling the output from the previous layer. However, we know that different layers have different receptive fields to the original image. For example, the initial layers have a smaller receptive field when compared to the final layers, which have a larger receptive field. Here, we will learn how SSD leverages this phenomenon to come up with a prediction of bounding boxes for images.
```

The workings behind how SSD helps overcome the issue of detecting objects with different scales is as follows: 

- We leverage the pre-trained VGG network and extend it with a few additional layers until we obtain a 1 x 1 block. 

- Instead of leveraging only the final layer for bounding box and class predictions, we will leverage all of the last few layers to make class and bounding box predictions. 

- In place of anchor boxes, we will come up with default boxes that have a specific set of scale and aspect ratios. 

- Each of the default boxes should predict the object and bounding box offset just like how anchor boxes are expected to predict classes and offsets in YOLO.

The network architecture of SSD is as follows:

![[Pasted image 20231016170834.png]]

We can see that after the last VGG-16 layer (conv5_3), the network is then extending by passing it through few more layers. Each layer attaches itself to a set amount of default box. We obtain a bounding box offset and class prediction for each cell and each default box.

In total, we get 8732 bounding boxes from all the feature maps throughout our layers.

![[Pasted image 20231016172826.png]]

The default boxes are determined on size and aspect ratio for each layer. The size factor is gradually increased as we go deeper in the network:

![[Pasted image 20231016172131.png]]


Possible aspect ratios are determined as follows:
![[Pasted image 20231016172219.png]]

![[Pasted image 20231016172240.png]]


When preparing our dataset, the default boxes that have an IoU greater than a threshold are considered positive matches.

#### Loss function

![[Pasted image 20231016172430.png]]

![[Pasted image 20231016172458.png]]

The bounding boxes from our feature maps (all 8732 of them) are used to compute the loss with the module `MultiBoxLoss`.

