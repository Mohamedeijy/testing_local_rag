Reference: [[Modern Computer Vision with PyTorch]]
Subjects: [[Deep Learning]]
Tags: #neuralnetwork

Note the the bias terms *w00*, *w01* are excluded in these operations to simplify the process.
### 1. Calculating the hidden layer unit values
Find the optimal weights of diagram:
![[Pasted image 20230901130314.png]]

	There are 9 (6 in the first hidden layer and 3 in the second) floats that we need to find, so that when the input is (1,1), the output is as close to (0) as possible.

Random selection of initial weights:
![[Pasted image 20230901130803.png]]

Hidden layer's unit values before activation:
![[Pasted image 20230901130848.png]]
![[Pasted image 20230901130941.png]]

### 2. Applying the activation function

Activation function allows to model a complex relation between input and output, that cannot be represented through the linear operations pre-activation. Different activations exist: 
- Sigmoid ![[Pasted image 20230901131227.png]]
- ReLU ![[Pasted image 20230901131236.png]]
- Tanh activation ![[Pasted image 20230901131245.png]]
- Linear activation (just the linear operations) ![[Pasted image 20230901131306.png]]

Book references each function visually: 
![[Pasted image 20230901131403.png]]

Using sigmoid activation, we obtain these values:
![[Pasted image 20230901131551.png]]

#### 3. Calculating the output layer value

Current network:
![[Pasted image 20230901131703.png]]

Sum of the products of weighs and previous hidden layer value:
![[Pasted image 20230901131907.png]]

Because values are random, the result is far from the expected output (0). Feedforward process is concluded by calculating **loss values (alternatively called cost functions)**.

Two scenarios are possible:
1. Output value is continuous
2. Output value to predict is categorical


##### 1. Continuous prediction

Typical loss value is** mean squared error (MSE)**.
![[Pasted image 20230901132450.png]]

	In the preceding equation,yi is the actual output.y^i is the prediction computed by the
	neural network ηθ(whose weights are stored in the form of ), where its input is , and m
	is the number of rows in the dataset.

So here the loss we need to minimize is:
![[Pasted image 20230901132845.png]]

##### 2. Categorical prediction

We typically use **[[Cross-entropy loss function]]**.

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
