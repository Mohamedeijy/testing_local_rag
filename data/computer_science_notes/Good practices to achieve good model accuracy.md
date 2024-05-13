---
References: "[[Modern Computer Vision with PyTorch]]"
subject:
  - "[[Deep Learning]]"
tags:
  - deeplearning
  - training
---
#### Scaling a dataset to improve model accuracy

Before scaling the variables:
![[Pasted image 20230913152420.png]]

After scaling the variables:
![[Pasted image 20230913152449.png]]

A nice pointer to understand why:
![[Pasted image 20230913152844.png]]

```
In the left-hand table, we can see that when the weight values are more than 0.1, the Sigmoid value does not vary with an increasing (changing) weight value. Furthermore, the Sigmoid value changed only by a little when the weight was extremely small; the only way to vary the sigmoid value is by having the weight change to a very, very small amount. However, the Sigmoid value changed considerably in the right-hand table when the input value was small. The reason for this is that the exponential of a large negative value (resulting from multiplying the weight value by a large number) is very close to 0, while the exponential value varies when the weight is multiplied by a scaled input, as seen in the right-hand table.
```

#### Understanding the impact of varying the batch size

In our example, when choosing a batch size of 32, we get:
![[Pasted image 20230913163401.png]]

Switching to a batch size of 10,000 give us these results:
![[Pasted image 20230913163436.png]]

```
Here, we can see that the accuracy and loss values did not reach the same levels as that of the previous scenario, where the batch size was 32, because the time weights are updated fewer times when the batch size is 32 (1875). In the scenario where the batch size is 10,000, there were six weight updates per epoch since there were 10,000 data points per batch, which means that the total training data size was 60,000.
```

#### Understanding the impact of varying the loss optimizer

In our example, using the SGD ([[Stochastic gradient descent]]) optimizer:
![[Pasted image 20230913165145.png]]

When switching with the Adam optimizer:
![[Pasted image 20230913165423.png]]


#### [[Understanding the impact of varying the learning rate]]


#### Building a deeper neural network

![[Pasted image 20230913174412.png]]

```
- The model was unable to learn as well as when there were no hidden layers. 

- The model overfit by a larger amount when there were two hidden layers compared to one hidden layer (the validation loss is higher in the model with two layers compared to the model with one layer).
```

#### [[Batch normalization]]

#### Impact of adding dropout (reduce overfitting)

Whenever  `loss.dropout()` is calculated, every weight of the network is updated. Certain parameters thus may be fine-tuned for the training data more than others. 

Dropout is the deactivation of a randomly selected set of weights for an iteration.
Because this mechanism can only be deployed during training, dropout is deactivated during evaluation and `the weights will be downscaled automatically during prediction (evaluation) to adjust for the magnitude of the weights (since all the weights are present during prediction time)`. This is why we select either `model.train()` or `model.eval()` when doing either task, otherwise the results would be unexpected, see example below:
![[Pasted image 20230914154509.png]]

To add a Dropout in our network, we insert the command before an activation where we want to specify that only a fixed percentage of the weights won't be activated:
![[Pasted image 20230914154929.png]]

This in our example, results in less overfitting, as the difference between our training and validation accuracy is lesser than before.

#### Impact of [[Regularization]] (reduce overfitting)
