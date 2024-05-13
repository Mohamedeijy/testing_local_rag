---
References: "[[Modern Computer Vision with PyTorch]]"
Subjects: "[[Deep Learning]]"
tags:
  - deeplearning
  - training
---
When the input values are really small, slight variations in the sigmoid output are the translation of very drastic weight changes:
![[Pasted image 20230913174959.png]]

```
Additionally, in the 'Scaling the input data' section, we saw that large input values have a negative effect on training accuracy. This suggests that we can neither have very small nor very big values for our input.

Along with very small or very big values in input, we may also encounter a scenario where the value of one of the nodes in the hidden layer could result in either a very small number or a very large number, resulting in the same issue we saw previously with the weights connecting the hidden layer to the next layer.
```

Batch normalization allows us to normalize the value at each node, just like when scaling our input values.

`Typically, all the input values in a batch are scaled as follows:`
![[Pasted image 20230914151651.png]]

By subtracting each data point from the batch mean and then dividing it by the batch variance, we have normalized all the data points of the batch at a node to a fixed range. While this is known as hard normalization, by introducing the γ and β parameters, we are letting the network identify the best normalization parameters.

```
Batch normalization helps considerably when training deep neural networks. It helps us avoid gradients becoming so small that the weights are barely updated.
```

