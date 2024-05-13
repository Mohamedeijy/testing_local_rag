---
References:
  - https://medium.com/mlearning-ai/developing-a-basketball-minimap-for-player-tracking-using-broadcast-data-and-applied-homography-433183b9b995
  - https://webthesis.biblio.polito.it/15863/1/tesi.pdf
  - https://openaccess.thecvf.com/content/WACV2022/papers/Shi_Self-Supervised_Shape_Alignment_for_Sports_Field_Registration_WACV_2022_paper.pdf
  - https://openaccess.thecvf.com/content/WACV2021/papers/Nie_A_Robust_and_Efficient_Framework_for_Sports-Field_Registration_WACV_2021_paper.pdf
Subjects: 
tags:
  - project
  - computervision
---
# Overall characteristics

# Reference read-throughs

## [Developing a Basketball Minimap for Player Tracking using Broadcast Data and Applied Homography](https://medium.com/mlearning-ai/developing-a-basketball-minimap-for-player-tracking-using-broadcast-data-and-applied-homography-433183b9b995)

The medium article that started it all. It relies on detectron2 to execute object detection on an image of the court. And to only allow for the detection of the people on the court, it sets a "manual" method of applying homography on one of two parallelepiped supposed to represent each side of the court, with a third one for the middle. Initial players and their position are set, a function searches for the closest player of a the previous frame to identify them in the next one. Those position are put on minimap to visualize positions and movement.

![[Pasted image 20231124150841.png]]
## [Computer vision for detecting and tracking players in basketball videos, Sara Battelini (Politecnico de Torino) (2020)](https://webthesis.biblio.polito.it/15863/1/tesi.pdf)

The objective at the center of this thesis is **Multiple Object Tracking (MOT)**, the task of following the trajectory of multiple objects in a sequence of images (frames of a video usually).
The standard approach of MOT is as quoted:
```
• Detection stage: this step is performed by an object detection algorithm, which will output a set of bounding boxes corresponding to the desired objects; 

• Feature extraction/motion prediction stage: the detections are analysed to extract features regarding appearance, motion, interaction. Optionally, a motion predictor predicts the next position of each tracked target; 

• Affinity stage: extracted features and motion predictions are used to compute a similarity/distance score between pairs of detections and/or tracklets; 

• Association stage: the similarity/distance scores are used to associate detections into tracks.
```

As seen above, MOT is a task that is essential to the functioning of JumperCV. Battelini's thesis also dives into court detection and homography to achieve functional player tracking.

Battelini's course of action is as follows:
- Search and cleaning of a dataset
- Network comparison (question of transfer learning or fine-tuning was in the mix but low quality dataset made it detrimental to network performance)
- Court detection and homography algorithm

## [A Robust and Efficient Framework for Sports-Field Registration (2021)](https://openaccess.thecvf.com/content/WACV2021/papers/Nie_A_Robust_and_Efficient_Framework_for_Sports-Field_Registration_WACV_2021_paper.pdf)


This paper presents research done by Amazon Prime Video research. It consists of a framework capable of registering sports-fields with real time broadcast video.
It executes keypoint based homography on a sports field, but instead of training a network to recognize a sparse amount of them on a field, they created a grid of points distributed uniformly over the entirety of the terrain. 
Coupled with those keypoints is a set of dense features extracted from the terrain, "defined as the normalized distance-map of each pixel in the field template to its nearest line and key region (e.g. yard-line numbers in American football)".

![[Pasted image 20231124143651.png]]

A first estimation of the homography and the dense features is made through our network, and then the homography is refined with additional information from the dense features.

Details on the structure of the neural network and it's composition can be found on the paper. The principal aspect of this research that needs attention is the grid of keypoints from which the estimation is based. This data is necessarily a set of ground truths presenting themselves as images of the field taken with a certain orientation and zoom of the camera, and then the grid laid on top of them with the correct homography computed. This type of annotated data is not available to the public (aside from the World Cup Soccer dataset), and I do not have the capacity and time to create such a detailed dataset. **It is important to reference [this paper](https://openjournals.uwaterloo.ca/index.php/vsl/article/view/3542) presenting a tool to create ground truths in the form of homography annotations with broadcast video.** It is, however, clear that keypoint detection is the way to go to perform homography on our basketball court.

## [Sports Field Registration via Keypoints-aware Label Condition (2022)](https://openaccess.thecvf.com/content/CVPR2022W/CVSports/papers/Chu_Sports_Field_Registration_via_Keypoints-Aware_Label_Condition_CVPRW_2022_paper.pdf)

The method used on this one for ground truth creation builds on the one established for the previous research paper (as it is referenced).
## [Real-Time Camera Pose Estimation for Sports Fields (2020)](https://arxiv.org/pdf/2003.14109.pdf)



## [Self-Supervised Shape Alignment for Sports Field Registration (2022)](https://openaccess.thecvf.com/content/WACV2022/papers/Shi_Self-Supervised_Shape_Alignment_for_Sports_Field_Registration_WACV_2022_paper.pdf)

This paper is mostly focused on the homography regression part. What makes its approach distinct is that it does not just use the normal image (ImA) and has the homographied image (ImB) as output. It also uses the edge image of ImB as a second input.

This made me realize that it was better to do homography regression as opposed to keypoint estimation. Not only because I will always have only a portion of the keypoints in an image, but also because my court image already is a model through which I can generate coordinates. I was always thinking "Yeah it's nice to have the homography but if I can't extract the borders of the court - thus its coordinates, it's useless.", but it was misguided. If I take my original broadcast image and manage to do homography estimation on it, I can use the inverse of that homography on the court model, and every position of the players is going to be the position on that homographied court.   

**One important note: "Note, to perform the inference with video, we could skip the model M(0) for subsequent frames, and use the homography from last frame to do the refinement."**

This paper also has a good approach to reduce overfitting (Sections 3.2 - Training sample generation and 5.3 - Experiments)
## [Optimizing Through Learned Errors for Accurate Sports Field Registration](https://arxiv.org/pdf/1909.08034.pdf)

This is it. This is the one.
This study uses one DNN to estimate a homography H(0) (with ground truth homographies in training), and then second one to minimize error and do refinments on H(10).
![[Pasted image 20240113152654.png]]

With further reading, my thinking was that it was predicting a transformation matrix, but what is actually being predicted is the 8 value vector of 4 points projected onto the warped template: 
![[Pasted image 20240222191245.png]]
My issue with this write-up is that I don't fully grasp the coordinate system they went with. "0.1" feels like a typo when you look at the actual figure illustrating the process:
![[Pasted image 20240222191354.png]]

Otherwise, I do not mind this difference in value predicted. One might say that it's easier to process when the points being projected are always at the same position within the image.

**Addendum**: So I guess 0.1 refers to the mentioned three-fifths (3/5 \* 1 - 0.5 = 0.1) so the range of value of the coordinates are -+0.5;-+0.5). Makes sense. At first I just didn't get why it wasnt easier to make the coordinate ranges 0-1 as to have the origin as the bottom left corner of the image, but it seems like having the origin (0,0) as one of the points to compute our homography can lead to issues. Here is me explaining this to myself in this little sketch:
![[Pasted image 20240226182000.png]]
# Roadmap

- ### Make a post on r/computervision to validate my roadmap and look for info on homography annotation tools and stuff.

- ### Complete the source list by reading the last 3 papers shown in the "Reference read-throughs" section and writing down valuable observation or things to back to.

- ### Find or create an NBA dataset and apply homography over keypoints to create my own ground truth.
	- Test for homography matrices to get the hang of it.
	- Find the correct tool for annotation.
	- Check if NBA footage is usable for training for an open license project.
	- Assemble images for annotation.
	- Annotate homographies with set amount of keypoints visible on an NBA court minimap.

- ### Find the correct network architecture to train for either keypoint detection or homography estimation.

- ### Benchmark available object detection models to use for player detection on the homographied results, this requires establishing evaluation methods. 

- ### Implement the affinity and association stages (as seen in the Politecnico de Torino paper) to assure tracking from one frame to another.

- ### Construct complete pipeline from homography to detected players, as to have a working set of exhibit videos.

- ### Extract position data from detected players and apply over basketball court minimap.

- ### Test methods on liking position data to shooting percentage, and displaying it under bounding box position.

- ### Construct a README and render public on GitHub.

- ### Research on player identification methods (jersey color, jersey number, initial parameters, team data).

- ### ???

- ### Profit.

# New CV concepts

## Hough transform

A Hough Transform is a computer vision/digital image processing technique used to detect shapes like lines, circles or ellipses. 
In the case of a standard edge detector, we get lines that can be broken or incomplete because of missing pixels or bad parameters; whereas a Hough Transform allows for continuity of curves. It consists of a feature extraction on the image where the number of solution cases does not need to be provided.
![[Pasted image 20231122144643.png]]
 It does so by using a voting scheme on the parameter space of each point. Cf. [this video](https://www.youtube.com/watch?v=XRBc_xkZREg&ab_channel=FirstPrinciplesofComputerVision) for reference.
## Homography

A homography is a transformation matrix that takes an image from one plane to another plane, through a point of projection. 
Given a set of matching features/points between 2 images, we can compute the homography H that best takes us from image 1 to image 2, making it possible to warp the coordinates of image 1 to that of image 2 or vice-versa.

![[Pasted image 20231122153452.png]]
![[Pasted image 20231122153831.png]]

The problem is solved with a method called Constrained Least Squares.
![[Pasted image 20231122154154.png]] 

## RANSAC algorithm

When doing homography given two images, we recognize that not all the pairs used to compute the transformation matrix are valid. We thus have to deal with the issue of inliers ( valid pairs) and outliers (invalid matches). This is where the RANSAC (RANdom SAmple Consensus) algorithm comes into play. 
RANSAC is general algorithm that can be used on any outlier problem.
1. Randomly chooses S minimum samples from our data.
2. Fit the model to the randomly chosen data (with homography, take 4 pairs of matching points and compute the homography).
3. Count the number M of data points (inliers) that fit the model within a set margin of error/threshold Epsilon.
4. Repeat N times.
5. Choose the model that has the largest amount of inliers M.

## SIFT
Or scale-invariant feature transform, is an algorithm used in CV to detect similar elements within two different images.
![[Pasted image 20240113153023.png]]

# Dataset
# Possibles networks/algorithms

## Court detection/homography 

## Object detection

## Tracking from bounding boxes

# Testing history

## Homography testing in opencv

Took a broadcast image and a court image and annotated points for both. Then used `cv2.findHomography` to generate a homography matrix from one to the other. 
I wanted to handle the image warping and homography for the first time to get familiar with the process. I also displayed the keypoints created on original and homographied images to vizualize.

![[Pasted image 20240113145714.png]]

![[Pasted image 20240113145721.png]]

![[Pasted image 20240113145729.png]]

### NEXT TASK: 
- After reading "Optimizing Through Learned Errors for Accurate Sports Field Registration" to completion, I need to choose the correct format for my networks. 
- I need to establish a vague architecture based on the paper (probably follow the 2 models pipeline they use).
- After that, I need to write an algo with OpenCV to loop over my broadcast images and:
	- Create multiple points by clicking over a broadcast image.
	- Displaying the NBA court model on the side and clicking the exact same number of points (if every point for both can be numbered it would be great)
	- Create a homography for both sets of points.
	- Add the homography to the corresponding annotation.

# Results