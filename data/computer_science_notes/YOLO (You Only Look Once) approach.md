---
References: "[[Modern Computer Vision with PyTorch]]"
Subjects: "[[Deep Learning]]"
tags:
  - computervision
  - deeplearning
  - objectdetection
  - cnn
---
Original paper: [[1506.02640] You Only Look Once: Unified, Real-Time Object Detection (arxiv.org)](https://arxiv.org/abs/1506.02640)

```
You Only Look Once (YOLO) and its variants are one of the prominent object detection algorithms. In this section, we will understand at a high level how YOLO works and the potential limitations of R-CNN-based object detection frameworks that YOLO overcomes.
```

The issue with a Faster R-CNN network is that at the level of the fully connected layer, only the region proposal's RoI pooling is passed (meaning that the prediction does not have the full picture). This results in our model having to guess the bounding box around the object if the region proposal does not encompass it.

YOLO is in different in that way as it looks at the whole image when predicting th ebounding box.

Furthermore, Faster R-CNN is composed of two networks (the RPN and the classifier module after it), which impedes on its inference speed. YOLO also improves upon this problem by having a single network.

# Working details

The initial intuition for YOLO is the idea that we want to split our image into an **S** x **S**  grid. The idea being that each cell will output a prediction, with a corresponding bound box.

![[Pasted image 20231016144132.png]]
The cell responsible for identifying the object being the one that contains the center of the ground truth of the object:

![[Pasted image 20231016144141.png]]

Each output and label for a set object will be relative to the cell. Meaning that our feature vector will correspond to a cell (our **y** here), and will contain the **bx** and **by** coordinates  of the ground truth **relative to the cell**, and **bw** and **bh** the sizing of our ground truth. It is important to note that those two values can be larger than the sizing of our cell, in the case of the object being larger and exceeding its boundaries. **pc** refers to the probability of the object being present (0 or 1) within the cell. The remaining **c** values are the one-hot encoded vector representing the class of appartenance of the object.

![[Pasted image 20231016144548.png]] ![[Pasted image 20231016144614.png]]


Our predictions will look similar but instead of just one bounding box being present, we make **two anchor boxes**, differing in heigh/width ratio (one wide, one tall) and we presume that these two boxes will specialize based on the type of object. This also allows to avoid the case where two object centers are present within the same cell.

![[Pasted image 20231016150129.png]]


The model architecture is gonna be 

![[Pasted image 20231016155305.png]]

# Loss function


```
When calculating the loss associated with the model, we need to ensure that we do not calculate the regression loss and classification loss when the objectness score is less than a certain threshold (this corresponds to the cells that do not contain an object). Next, if the cell contains an object, we need to ensure that the classification across different classes is as accurate as possible. Finally, if the cell contains an object, the bounding box offsets should be as close to expected as possible. However, since the offsets of width and height can be much higher when compared to the offset of the center (as offsets of the center range between 0 and 1, while the offsets of width and height need not), we give a lower weightage to offsets of width and height by fetching a square root value.
```

![[Pasted image 20231016155516.png]]
![[Pasted image 20231016155527.png]]

The overall loss is a sum of classification and regression loss values, each loss is some form of squared error.

```
Note that the loss function only penalizes classification error if an Object is present in that grid cell (hence the conditional class probability discussed earlier). It also only penalizes bounding box coordinate error if that predictor is
"responsible" for the ground truth box (i.e. has the highest IOU of any predictor in that grid cell).
```
