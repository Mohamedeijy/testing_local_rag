---
References: "[[Modern Computer Vision with PyTorch]]"
Subjects: "[[Deep Learning]]"
tags:
  - computervision
  - neuralnetwork
  - cnn
---
## Convolution

`Let's assume we have two matrices we can use to perform convolution. Here is Matrix A:`
![[Pasted image 20230914164339.png]]

`Here is Matrix B:`
![[Pasted image 20230914164424.png]]

Performing convolution basically means sliding Matrix B (the smaller matrix, typically called a **filter** or a **kernel**) over Matrix A, performing element to element multiplication between the two at each time, as follows:

1. Multiply {1,2,5,6} of the bigger matrix by {1,2,3,4} of the smaller matrix: 
	1* 1 + 2* 2 + 5* 3 + 6* 4 = 44 
2. Multiply {2,3,6,7} of the bigger matrix by {1,2,3,4} of the smaller matrix: 
	2* 1 + 3* 2 + 6* 3 + 7* 4 = 54 
3. Multiply {3,4,7,8} of the bigger matrix by {1,2,3,4} of the smaller matrix: 
	3* 1 + 4* 2 + 7* 3 + 8* 4 = 64 
4. Multiply {5,6,9,10} of the bigger matrix by {1,2,3,4} of the smaller matrix: 
	5* 1 + 6* 2 + 9* 3 + 10* 4 = 84 
5. Multiply {6,7,10,11} of the bigger matrix by {1,2,3,4} of the smaller matrix: 
	6* 1 + 7* 2 + 10* 3 + 11* 4 = 94 
6. Multiply {7,8,11,12} of the bigger matrix by {1,2,3,4} of the smaller matrix: 
	7* 1 + 8* 2 + 11* 3 + 12* 4 = 104 
7. Multiply {9,10,13,14} of the bigger matrix by {1,2,3,4} of the smaller matrix: 
	9* 1 + 10* 2 + 13* 3 + 14* 4 = 124 
8. Multiply {10,11,14,15} of the bigger matrix by {1,2,3,4} of the smaller matrix: 
	10* 1 + 11* 2 + 14* 3 + 15* 4 = 134 
9. Multiply {11,12,15,16} of the bigger matrix by {1,2,3,4} of the smaller matrix: 
	11* 1 + 12* 2 + 15* 3 + 16* 4 = 144

Giving us the resulting matrix:
![[Pasted image 20230914164959.png]]

## Filters

As previously mentioned, the smaller matrix is usually called a filter or a kernel. The model learns the optimal weight in those filters through training. In general, the more filters there are, the more features can be learned through these filters.


```
In the previous section, we learned that when we convolved one filter that has a size of 2 x 2 with a matrix that has a size of 4 x 4, we got an output that is 3 x 3 in dimension. However, if 10 different filters multiply the bigger matrix (original image), the result is 10 sets of the 3 x 3 output matrices.
```

```
Furthermore, in a scenario where we are dealing with color images where there are three channels, the filter that is convolving with the original image would also have three channels, resulting in a single scalar output per convolution. Also, if the filters are convolving with an intermediate output – let's say of 64 x 112 x 112 in shape – the filter would have 64 channels to fetch a scalar output. 

In addition, if there are 512 filters that are convolving with the output that was obtained in the intermediate layer, the output post convolution with 512 filters would be 512 x 111 x 111 in shape.
```
![[Pasted image 20230914170456.png]]

## Strides

In our previous example, convoluting Matrix B over Matrix A with a stride of 2 as follows:

1. {1,2,5,6} of the bigger matrix is multiplied by {1,2,3,4} of the smaller matrix: 
	1* 1 + 2* 2 + 5* 3 + 6* 4 = 44 
2. {3,4,7,8} of the bigger matrix is multiplied by {1,2,3,4} of the smaller matrix: 
	3* 1 + 4* 2 + 7* 3 + 8* 4 = 64 
3. {9,10,13,14} of the bigger matrix is multiplied by {1,2,3,4} of the smaller matrix: 
	9* 1 + 10* 2 + 13* 3 + 14* 4 = 124 
4. {11,12,15,16} of the bigger matrix is multiplied by {1,2,3,4} of the smaller matrix: 
	11* 1 + 12* 2 + 15* 3 + 16* 4 = 144

Resulting in the following 2 by 2 matrix:
![[Pasted image 20230914171034.png]]

The preceding output has a lower dimension compared to the scenario where the stride was 1 (where the output shape was 3 x 3) since we now have a stride of 2.

This results in a partial loss of information and can affect the possibility of us adding the output of the convolution operation to the original image (residual addition).


## Padding

```
In the preceding case, we could not multiply the leftmost elements of the filter by the rightmost elements of the image. If we were to perform such matrix multiplication, we would pad the image with zeros. This would ensure that we can perform element to element multiplication of all the elements within an image with a filter.
```
![[Pasted image 20230914171413.png]]

## Pooling

Pooling concentrates information in a small patch. For example, a max pooling with a sliding window of size 2x2 and a stride of 2 (effectively diving the matrix into 4 quadrants):

![[Pasted image 20230914171725.png]]

Into:
![[Pasted image 20230914171739.png]]

## Flattening through the fully connected layer

The overall flow of a CNN is as follows:
![[Pasted image 20230914171850.png]]

```
Convolution and pooling help in fetching a flattened layer that has a much smaller representation than the original image.
```
After the feature learning process is done and the intermediate data is of a reduced dimension, it is flattened. The rest is a sequence of hidden layers like in the classic ANN example, and then an output layer.

