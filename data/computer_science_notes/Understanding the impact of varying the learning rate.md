---
References: "[[Modern Computer Vision with PyTorch]]"
subject:
  - "[[Deep Learning]]"
tags:
  - deeplearning
  - training
---
#### Impact of the learning rate on a scaled dataset

- **High learning rate**
![[Pasted image 20230913170206.png]]
(mediocre accuracy and high loss that doesn't fluctuate much). The model is unable to be trained.

- **Medium learning rate**
![[Pasted image 20230913170439.png]]

- **Low learning rate**
Because of low learning rate, there needs to be more epochs (so the number is set to 100).
![[Pasted image 20230913170526.png]]

It took the low learning rate 100 epochs to reach 89%, whereas it took around 10 for a medium learning rate.

#### Analyzing parameter distribution across layers for different learning rates

![[Pasted image 20230913171431.png]]

	- Larger distribution when learning rate is higher
	- Tends to overfit more the larger the disctribution

#### Impact of the learning rate on a non-scaled dataset
![[Pasted image 20230913171730.png]]

With non-scaled data training is not attainable with high and moderate learning rates. A very low learning rate achieves learning over a high number of epochs but overfits quickly on the training data. **Why ?**

![[Pasted image 20230913172058.png]]

```
Here, we can see that when the model accuracy was high (which is when the learning rate was 0.00001), the weights had a much smaller range (typically ranging between -0.05 to 0.05 in this case) compared to when the learning rate was high. The weights can be tuned toward a small value since the learning rate is small. 

Note that the scenario where the learning rate is 0.00001 on a non-scaled dataset is equivalent to the scenario of the learning rate being 0.001 on a scaled dataset. This is because the weights can now move toward a very small value (because gradient * learning rate is a very small value, given that the learning rate is small).
```

### Learning rate annealing (reduce the learning rate automatically when the model starts overfitting)

```
So far, we have initialized a learning rate, and it has remained the same across all the epochs while training the model. However, initially, it would be intuitive for the weights to be updated quickly to a near-optimal scenario. 
From then on, they should be updated very slowly since the amount of loss that gets reduced initially is high and the amount of loss that gets reduced in the later epochs would be low. This calls for having a high learning rate initially and gradually lowering it later on as the model achieves near-optimal accuracy. 
This requires us to understand when the learning rate must be reduced. 

One potential way we can solve this problem is by continually monitoring the validation loss and if the validation loss does not decrease (let's say, over the previous x epochs), then we reduce the learning rate.
```

In PyTorch, the tool used to reduce the learning rate if the validation does not decrease in the previous x epochs is the Scheduler with the `lr_scheduler`.

![[Pasted image 20230913174041.png]]