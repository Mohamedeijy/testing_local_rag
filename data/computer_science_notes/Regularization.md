---
References: "[[Modern Computer Vision with PyTorch]]"
Subjects: "[[Deep Learning]]"
tags:
  - deeplearning
  - training
---

```
Apart from the training accuracy being much higher than the validation accuracy, one other feature of overfitting is that certain weight values will be much higher than the other weight values. High weight values can be a symptom of the model learning very well on training data (essentially, a rot learning on what it has seen).
```

While dropout is used to diminish the frequency at which a weight is updated, regularization is a technique in which we penalize the network for having high weight values. This is supposed to achieve a lower loss on training data and lower weight values. 

### L1 Regularization

L1 regularization is calculated as follows:
![[Pasted image 20230914160340.png]]

```
The first part of the preceding formula refers to the categorical cross-entropy loss that we have been using for optimization so far, while the second part refers to the absolute sum of the weight values of the model. Note that L1 regularization ensures that it penalizes for the high absolute values of weights by incorporating them in the loss value calculation. 

Λ refers to the weightage that we associate with the regularization (weight minimization) loss.
```

L1 regularization is implemented while training the model (`torch.norm(param,1) provides the absolute value`):

![[Pasted image 20230914160749.png]]

He the weightage used is 0.0001, which is very small.

### L2 Regularization

L2 regularization is calculated as follows:
![[Pasted image 20230914161058.png]]

L2 regularization is implemented while training the model:
![[Pasted image 20230914161518.png]]

```
Note that in the preceding code, the regularization parameter, (0.01), is slightly higher than in L1 regularization since the weights are generally between -1 to 1 and a square of them would result in even smaller values. Multiplying them by an even smaller number, as we did in L1 regularization, would result in us having very little weightage for regularization in the overall loss calculation.
```

### Comparing weight distribution between regularization and no regularization

![[Pasted image 20230914161636.png]]

We can see that weight distribution is smaller when we regularize the network. This is in accordance with the goal of keeping our weight values from updating for edge cases.