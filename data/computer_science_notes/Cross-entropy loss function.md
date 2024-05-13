---
References: "[[Modern Computer Vision with PyTorch]]"
Subjects: "[[Deep Learning]]"
tags:
  - loss_function
  - evaluation
---

	When the variable to predict has two distinct values within it, the loss function is
	binary cross-entropy.

Binary cross-entropy:
![[Pasted image 20230901133047.png]]

	y is the actual value of the output, p is the predicted value of the output, and m is
	the total number of data points.

Categorical cross-entropy:
![[Pasted image 20230901133354.png]]

	y is the actual value of the output, p is the predicted value of the output, m is the
	total number of data points, and C is the total number of classes.


Example provided by the book: 

	A simple way of visualizing cross-entropy loss is to look at the prediction matrix
	itself. Say you are predicting five classes – Dog, Cat, Rat, Cow, and Hen – in an image
	recognition problem. The neural network would necessarily have five neurons in the
	last layer with softmax activation (more on softmax in the next section). It will be
	thus forced to predict a probability for every class, for every data point. Say there
	are five images and the prediction probabilities look like so (the highlighted cell in
	each row corresponds to the target class):

	![[Pasted image 20230901133936.png]]

	Note that each row sums to 1. In the first row, when the target is Dog and the
	prediction probability is 0.88, the corresponding loss is 0.128 (which is the negative
	of the log of 0.88). Similarly, other losses are computed. As you can see, the loss
	value is less when the probability of the correct class is high. As you know, the
	probabilities range between 0 and 1. So, the minimum possible loss can be 0 (when the
	probability is 1) and the maximum loss can be infinity when the probability is 0. The
	final loss within a dataset is the mean of all individual losses across all rows.
