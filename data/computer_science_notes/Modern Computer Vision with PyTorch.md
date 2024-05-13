---
annotation-target: Books/Computer Science/modern-computer-vision-with-pytorch-v-kishore-1--annas-archive.pdf
author:
  - V Kishore Ayyadevara
  - Yeshwanth Reddy
subject:
  - "[[Computer Vision]]"
  - "[[PyTorch]]"
tags:
  - book
  - computervision
---
## Outline

- Starts with the basics of neural networks and covers over 50 applications of computer vision. First, you will **build a neural network (NN) from scratch** using both NumPy, PyTorch, and then learn the **best practices of tweaking a NN's hyperparameters**.
  
- As we progress, you will **learn about CNNs**, **transfer-learning with a focus on classifying images**. You will also **learn about the practical aspects to take care of while building a NN** model. 
  
- Next, you will learn about **multi-object detection, segmentation, and implement them using R-CNN family, SSD, YOLO, U-Net, Mask-RCNN architectures**. You will then **learn to use the Detectron2 framework to simplify the process of building a NN for object detection and human-pose-estimation**. Finally, you will **implement 3-D object detection**. 
  
- Subsequently, you will **learn about auto-encoders and GANs with a strong focus on image manipulation and generation**. Here, you will **implement VAE, DCGAN, CGAN, Pix2Pix, CycleGan, StyleGAN2, SRGAN, Style-Transfer to manipulate images** on a variety of tasks.
  
- You will then learn to **combine NLP and CV techniques while performing OCR**, **Image Captioning**, **object detection with transformers**. Next, you will learn to **combine RL with CV techniques to implement a self-driving car agent**. Finally, you'll wrap up with **moving a NN model to production and learn conventional CV techniques using the OpenCV library**.

## Readthrough

### Preface


>%%
>```annotation-json
>{"created":"2023-08-29T16:52:54.621Z","text":"If I get to finish this by the end of September I'm already good.","updated":"2023-08-29T16:52:54.621Z","document":{"title":"modern-computer-vision-with-pytorch-v-kishore-1--annas-archive.pdf","link":[{"href":"urn:x-pdf:bde92396ca5535a68f5381fcf7b7e9b8"},{"href":"vault:/999 Attachments/Books/Computer Science/modern-computer-vision-with-pytorch-v-kishore-1--annas-archive.pdf"}],"documentFingerprint":"bde92396ca5535a68f5381fcf7b7e9b8"},"uri":"vault:/999 Attachments/Books/Computer Science/modern-computer-vision-with-pytorch-v-kishore-1--annas-archive.pdf","target":[{"source":"vault:/999 Attachments/Books/Computer Science/modern-computer-vision-with-pytorch-v-kishore-1--annas-archive.pdf","selector":[{"type":"TextPositionSelector","start":19059,"end":19392},{"type":"TextQuoteSelector","exact":"Next, you will learn about multi-object detection, segmentation, and implement themusing R-CNN family, SSD, YOLO, U-Net, Mask-RCNN architectures. You will thenlearn to use the Detectron2 framework to simplify the process of building a NN forobject detection and human-pose-estimation. Finally, you will implement 3-D objectdetection.","prefix":"care of while building a NN model.","suffix":"Subsequently, you will learn abo"}]}]}
>```
>%%
>*%%PREFIX%%areof while building a NN model.%%HIGHLIGHT%% ==Next, you will learn about multi-object detection, segmentation, and implement themusing R-CNN family, SSD, YOLO, U-Net, Mask-RCNN architectures. You will thenlearn to use the Detectron2 framework to simplify the process of building a NN forobject detection and human-pose-estimation. Finally, you will implement 3-D objectdetection.== %%POSTFIX%%Subsequently, you will learn abo*
>%%LINK%%[[#^ncqzjivzt6|show annotation]]
>%%COMMENT%%
>If I get to finish this by the end of September I'm already good.
>%%TAGS%%
>
^ncqzjivzt6


## Section 1, Fundamentals of Deep Learning for Computer Vision
### Chapter 1, Artificial Neural Network Fundamentals

#### Concept 

[[Artificial Neural Network (ANN)]]

We get to do an ANN from scratch (note for it [[ANN from scratch for MCVWPT book]]). The layout is:
1. Comparing AI and traditional machine learning 
2. Learning about the artificial neural network building blocks 
3. Implementing feedforward propagation 
4. Implementing backpropagation

#### 1. Comparing AI and traditional machine learning

Traditional ML : ML practitioner identifies and extracts features and then trains algo like a classifier to identify them.
	`therefore, the number of manual rules we'd need to create for the accurate classification of multiple types can be exponential, especially as images become more complex`

An AI tool like a neural network allows to do the feature extraction and the training in a single shot.
	`It does not require a human to come up with rules to classify an image, which takes away the majority of the burden traditional techniques impose on the programmer`

#### 2. Learning about the artificial neural network building blocks

ANN structure:
- Input layer
- Hidden layers
- Output layers

Zoom into a neuron inside the hidden layer:
![[Pasted image 20230901125401.png]]

Based *wi* the weights and *w0* the bias term, output value *a* is as follows:

![[Pasted image 20230901125602.png]]


#### 3. [[Implementing feedforward propagation]]

#### 4. [[Implementing backpropagation]]

Annotations de correction sur la section sur la backpropagation.

>%%
>```annotation-json
>{"created":"2023-09-01T12:56:38.173Z","text":"Correction provided in https://github.com/PacktPublishing/Modern-Computer-Vision-with-PyTorch/tree/master/Chapter01","updated":"2023-09-01T12:56:38.173Z","document":{"title":"modern-computer-vision-with-pytorch-v-kishore-1--annas-archive.pdf","link":[{"href":"urn:x-pdf:bde92396ca5535a68f5381fcf7b7e9b8"},{"href":"vault:/999 Attachments/Books/Computer Science/modern-computer-vision-with-pytorch-v-kishore-1--annas-archive.pdf"}],"documentFingerprint":"bde92396ca5535a68f5381fcf7b7e9b8"},"uri":"vault:/999 Attachments/Books/Computer Science/modern-computer-vision-with-pytorch-v-kishore-1--annas-archive.pdf","target":[{"source":"vault:/999 Attachments/Books/Computer Science/modern-computer-vision-with-pytorch-v-kishore-1--annas-archive.pdf","selector":[{"type":"TextPositionSelector","start":69724,"end":69778},{"type":"TextQuoteSelector","exact":"The predicted output value   is calculated as follows:","prefix":"twork is represented as follows:","suffix":"The hidden layer activation valu"}]}]}
>```
>%%
>*%%PREFIX%%twork is represented as follows:%%HIGHLIGHT%% ==The predicted output value   is calculated as follows:== %%POSTFIX%%The hidden layer activation valu*
>%%LINK%%[[#^th44fqufmsk|show annotation]]
>%%COMMENT%%
>Correction provided in https://github.com/PacktPublishing/Modern-Computer-Vision-with-PyTorch/tree/master/Chapter01
>%%TAGS%%
>
^th44fqufmsk


### Chapter 2, PyTorch Fundamentals

#### PyTorch tensor

PyTorch tensor summary:
![[Pasted image 20230901155638.png]]


#### Auto-gradient of tensor objects

. Defining a tensor object and also specifying that it requires a gradient to be calculated, allows you to get the gradient of a value based on the tensor's change:

![[Pasted image 20230901165710.png]]

In the image, the gradient of  `out` with respect to `x` calculated corresponds to what, intuitively, we think it would be for `out=∑x²`, being its derivative `2x`.

**All the subsequent manipulations were done on the notebooks inside my folder for the book**
### [[Saving and loading a PyTorch model]]


### Chapter 3, Building a Deep Neural Network with PyTorch

#### Training a neural network

```
To train a neural network, we must perform the following steps: 
1. Import the relevant packages. 
2. Build a dataset that can fetch data one data point at a time. 
3. Wrap the DataLoader from the dataset. 
4. Build a model and then define the loss function and the optimizer. 
5. Define two functions to train and validate a batch of data, respectively. 
6. Define a function that will calculate the accuracy of the data. 
7. Perform weight updates based on each batch of data over increasing epochs.
```

[2] When building a class that fetches the dataset, remember that it is derived from the Dataset class, and it needs 3 functions that are always defined : `__init__`, `__getitem__` and `__len__`.

[4] In the example where we create a model using `nn.Sequential()`, we do not create a softmax layer, even though cross-entropy loss typically expects outputs as probabilities (each row should sum to 1), because the CrossEntropyLoss function expects the unrestricted logits, the softmax is done internally.

#### [[Good practices to achieve good model accuracy]]

## Section 2, Object Classification and Detection

### Chapter 4, Introducing [[Convolutional Neural Networks]]

#### The problem with traditional deep neural networks

To highlight the limitations of a traditional deep neural network, we chose a random image within the data we were previously working with, where the highest prediction is, by default, "Trouser".

![[Pasted image 20230914163738.png]]

By rolling the image 5 pixel left and 5 pixels right, we can redo our prediction using the image slightly shifted in either direction.
In our results, we can see that, while there was no change in the content, the predicted class changed when the translation was beyond 2 pixels. The model is not trained to correctly recognize off-center content.

![[Pasted image 20230914163447.png]]

CNNs ([[Convolutional Neural Networks]]) address this major limitation.

#### [[Implementing data augmentation]]



### Chapter 5, Transfer Learning for Image Classification 

The workflow of transfer learning is as follows:

1. Normalize the input images, **normalized by the same mean and standard deviation that was used during the training of the pre-trained model**. 
2. Fetch the pre-trained model's architecture. Fetch the weights for this architecture that arose as a result of being trained on a large dataset. 
3. Discard the last few layers of the pre-trained model. 
4. Connect the truncated pre-trained model to a freshly initialized layer (or layers) where weights are randomly initialized. Ensure that the output of the last layer has as many neurons as the classes/outputs we would want to predict 
5. Ensure that the weights of the pre-trained model are not trainable (in other words, frozen/not updated during backpropagation), but that the weights of the newly initialized layer and the weights connecting it to the output layer are trainable.
6. Update the trainable parameters over increasing epochs to fit the model.

#### Understanding VGG16 architecture

```
VGG stands for Visual Geometry Group, which is based out of the University of Oxford, and 16 stands for the number of layers in the model.
```

![[Pasted image 20230921162908.png]]

```
In the preceding summary, the 16 layers we mentioned are grouped as follows:

{1,2},{3,4,5},{6,7},{8,9,10},{11,12},{13,14},{15,16,17},{18 ,19},{20,21},{22,23,24},{25,26},{27,28},{29,30,31,32},{33,3 4,35},{36,37,38],{39}
```

![[Pasted image 20230921162949.png]]

VGG19 gives slightly better accuracy than VGG16, and VGG11 slightly lower accuracy. By that token, increasing the amounts of layers on the VGG architecture should increase its performance. The truth is quite the contrary because of a few issues that arise:
- We have to learn a larger number of features. 
- Vanishing gradients arise. 
- There is too much information modification at deeper layers.

The ResNet architecture allows to address these issues by incorporating through the learning process the information of when to learn and when not to learn.

#### Understanding ResNet architecture

```
While building too deep a network, there are two problems. 

In forward propagation, the last few layers of the network have almost no information about what the original image was. 

In backpropagation, the first few layers near the input hardly get any gradient updates due to vanishing gradients (in other words, they are almost zero). 

To solve both problems, residual networks (ResNet) use a highway-like connection that transfers raw information from the previous few layers to the later layers. In theory, even the last layer will have the entire information of the original image due to this highway network.
And because of the skipping layers, the backward gradients will flow freely to the initial layers with little modification.
```

A residual block:

![[Pasted image 20230921163458.png]]

Not only are we extracting F(x) (value after passing through the weight layers), but we're also summing up the original value x.

ResNet18 has 18 layers in its architecture, a skip connection is made across the network every two layers:
![[Pasted image 20230921165427.png]]

#### Multi-regression with facial key point detection 

Multi-regression: predict multiple values given an image as input.

Facial key point detection:
![[Pasted image 20230921165820.png]]

This is a regression problem where we are predicting several continuous outputs.

#### Multi-task learning by implementing age estimation and gender classification

```
Multi-task learning is a branch of research where a single/few inputs are used to predict several different but ultimately connected outputs. 

For example, in a selfdriving car, the model needs to identify obstacles, plan routes, give the right amount of throttle/brake and steering, to name but a few. 
It needs to do all of these in a split second by considering the same set of inputs (which would come from several sensors)
```

In this use-case our goal is to predict age (continuous) and gender (categorical) in a single shot from an image.

To implement this multi-task learning, the model building process changes as follows:

- In the last part of the model, create two separate layers branching out from the preceding layer, where one layer corresponds to age estimation and the other to gender classification

- Ensure that you have different loss functions for each branch of output, as age is a continuous value (requiring an MSE or MAE loss calculation) and gender is a categorical value (requiring a cross-entropy loss calculation).

- Take a weighted summation of age estimation loss and gender classification loss. Minimize the overall loss through backpropagation.


### Chapter 6, Practical Aspects of Image Classification

#### Generating CAMs (class activation maps)


```
An example CAM is as follows, where we have the input image on the left and the pixels that were used to come up with the class prediction highlighted on the right:
```
![[Pasted image 20230928143404.png]]


CAMs are generated by calculating the gradients of a chosen activation layer (let's say the feature shape at a random convolution layer is 512 x 7 x 7) in relation to the output value representing a specific class. 
The mean (output shape is 512) of these gradients (the output gradient shape is 256 x 512 x 3 x 3, 3 being kernel size, 256 the input channels before convolution/activation) is computed.

Then the **weighted activation map** (the output shape is 512 x 7 x 7) is calculated by multiplying the gradient means by the activations channels. The output shape being *n-activation-layers x n-convolution-layer-width x n-convolution-layer-height*. The mean across all (512) channels is calculated and the resulting layer is upscaled to be overlayed on top of the original image

```
The following diagram from the paper Grad-CAM: Gradient-weighted Class Activation Mapping (https://arxiv.org/abs/1610.02391) pictorially describes the preceding steps:
```
![[Pasted image 20230928144930.png]]

#### Practical aspects to take care of during model implementation

##### Dealing with imbalanced data

```
Imagine a scenario where you are trying to predict an object that occurs very rarely within our dataset – let's say in 1% of the total images. For example, this can be the task of predicting whether an X-ray image suggests a rare lung infection. 

How do we measure the accuracy of the model that is trained to predict the rare lung infection? If we simply predict a class of no infection for all images, the accuracy of classification is 99%, while still being useless. A confusion matrix that depicts the number of times the rare object class has occurred and the number of times the model predicted the rare object class correctly comes in handy in this scenario. Thus, the right set of metrics to look at in this scenario is the metrics related to the confusion matrix.
```

The typical confusion matrix:
![[Pasted image 20230928152829.png]]
This a good tool (better than plain accuracy) to understand how accurate the model is.

In terms of training, cross-entropy does a good job assuring that the loss is high when misclassification is high. It is possible to encourage this by adding a weight to the class that occurs rarely. The other solutions are tools we've already seen: data augmentation, transfer learning, oversampling the rare class.

##### The size of the object within an image

```
Imagine a scenario where the presence of a small patch within a large image dictates the class of the image – for example, lung infection identification where the presence of certain tiny nodules indicates an incident of the disease. In such a scenario, image classification is likely to result in inaccurate results, as the object occupies a smaller portion of the entire image. Object detection comes in handy in this scenario (which we will study in the next chapter). 

A high-level intuition to solve these problems would be to first divide the input images into smaller grid cells (let's say a 10 x 10 grid) and then identify whether a grid cell contains the object of interest.
```

##### The difference between training and validation data

```
Imagine a scenario where you have built a model to predict whether the image of an eye indicates that the person is likely to be suffering from diabetic retinopathy. To build the model, you have collected data, curated it, cropped it, normalized it, and then finally built a model that has very high accuracy on validation images. However, hypothetically, when the model is used in a real setting (let's say by a doctor/nurse), the model is not able to predict well.
```

Multiple elements can be put into question with this example :

- Similarity/difference between data used for training and real images taken at the doctor's office :
	- If the training data is very curated with lots of preprocessing, whereas IRL data is not curated.
	- Device used to capture the images has a different resolution at the doctor's office
	- Both settings have different lighting conditions.

- Are the subjects (images) representative enough of the overall population?
	- Images are representative if they are trained on images of the male population but are tested on the female population, or if, in general, the training and real-world images correspond to different demographics.

- Is the training and validation split done methodically ?
	- Imagine a scenario where there are 10,000 images and the first 5,000 images belong to one class and the last 5,000 images belong to another class. When building a model, if we do not randomize but split the dataset into training and validation with consecutive indices (without random indices), we are likely to see a higher representation of one class while training and of the other class during validation.


In general, we need to ensure that the training, validation, and real-world images all have similar data distribution before an end user leverages the system.

##### The number of nodes in the flatten layer

```
Consider a scenario where you are working on images that are 300 x 300 in dimensions. Technically, we can perform more than five convolutional pooling operations to get the final layer that has as many features as possible. Furthermore, we can have as many channels as we want in this scenario within a CNN. Practically, though, in general, we would design a network so that it has 500–5,000 nodes in the flatten layer.
```


##### Image size

```
Let's say we are working on images that are of very high dimensions – for example, 2,000 x 1,000 in shape. When working on such large images, we need to consider the following possibilities: 

Can the images be resized to lower dimensions? Images of objects might not lose information if resized; however, images of text documents might lose considerable information if resized to a smaller size. 

Can we have a lower batch size so that the batch fits into GPU memory? Typically, if we are working with large images, there is a good chance that for the given batch size, the GPU memory is not sufficient to perform computations on the batch of images. 

Do certain portions of the image contain the majority of the information, and hence can the rest of the image be cropped?
```

##### Leveraging OpenCV utilities

In a scenario of having to deploy a model to production, less complexity is generally preferable. If a task can be replaced by an OpenCV module, it should be advantaged even if it is at the cost of a bit of accuracy. Building a model from scratch should be a consideration only if it gives a considerable boost in accuracy for our specific ouse-case.

### Chapter 7, [[Basics of Object Detection]]

### Chapter 8, [[Advanced Object Detection]]

### Chapter 9, [[Image Segmentation]]

