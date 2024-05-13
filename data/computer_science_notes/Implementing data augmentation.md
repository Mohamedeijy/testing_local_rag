---
References: "[[Modern Computer Vision with PyTorch]]"
Subjects: "[[Deep Learning]]"
tags:
  - deeplearning
  - training
---
In our example page 184 of [[Modern Computer Vision with PyTorch]], we made a CNN and redid a test where we roll the image of some trouser for 5 pixels left and right:

![[Pasted image 20230915143921.png]]

We can see that the model is much less vulnerable to translation compared to the classic ANN. However, after 5 pixels, the same issue arises. To prevent this, we need to use a technique called data augmentation.

#### Data augmentation

The `augmenters` class in the `imgaug` package has useful utilities for performing these augmentations.

Data augmentation can be done to simulate different types of scenario:

**Affine transformations:**
- Rotation
- Translation
- Scaling (zoom in/out)
- Shearing
For these types of transformation, the newly created pixels they result in can be set a certain way (constant color, reflection symmetry, etc) through the `mode` parameter of augmenters, and also it's color with the `cval` parameter. A range of values can also be given for a transformation:

```
from imgaug import augmenters as iaa

# rotate between 45 to the right and 45 to the left
aug = iaa.Affine(rotate=(-45,45), fit_output=True, cval=0, \ mode='constant')

plt.imshow(aug.augment_image(tr_images[0]), cmap='gray') plt.subplot(153)
```

![[Pasted image 20230915153637.png]]

**Changing the brightness:**

```
Imagine a scenario where the difference between the background and the foreground is not as distinct as we have seen so far. This means the background does not have a pixel value of 0 and that the foreground does not have a pixel value of 255. Such a scenario can typically happen when the lighting conditions in the image are different. If the background has always had a pixel value of 0 and the foreground has always had a pixel value of 255 when the model has been trained but we are predicting an image that has a background pixel value of 20 and a foreground pixel value of 220, the prediction is likely to be incorrect.
```

`Multiply` multiplies each pixels by a specified amount:

```
aug = iaa.Multiply(0.5) plt.imshow(aug.augment_image(tr_images[0]), cmap='gray', \ vmin = 0, vmax = 255) plt.title('Pixels multiplied by 0.5')
```
![[Pasted image 20230915154002.png]]

`Linearcontrast` adjusts each pixel based on this formula:
![[Pasted image 20230915154052.png]]


```
aug = iaa.LinearContrast(0.5) plt.imshow(aug.augment_image(tr_images[0]), cmap='gray', \ vmin = 0, vmax = 255) plt.title('Pixel contrast by 0.5')
```
![[Pasted image 20230915154112.png]]

**Blur the images:**

Blurred images is a realistic scenario that happens frequently. To stimulate this, we can use Gaussian blur using the `GaussianBlur` method from `augmenters`.

```
aug = iaa.GaussianBlur(sigma=1) plt.imshow(aug.augment_image(tr_images[0]), cmap='gray', \ vmin = 0, vmax = 255) plt.title('Gaussian blurring of image')
```
![[Pasted image 20230915154742.png]]

**Adding noise:**

Due to bad photography, images can be grainy. `Dropout` and `SaltAndPepper` are too useful methods used to add noise and simulate grain.

```
plt.figure(figsize=(10,10)) 
plt.subplot(121)

aug = iaa.Dropout(p=0.2) 
plt.imshow(aug.augment_image(tr_images[0]), cmap='gray', \ vmin = 0, vmax = 255) 
plt.title('Random 20% pixel dropout') 

plt.subplot(122) 
aug = iaa.SaltAndPepper(0.2) plt.imshow(aug.augment_image(tr_images[0]), cmap='gray', \ vmin = 0, vmax = 255) plt.title('Random 20% salt and pepper noise')
```

![[Pasted image 20230915155027.png]]

While `Dropout` sets a random pixel to zero value (rendering them dark), `SaltAndPepper` creates black and white pixels.

During training, after having selected the augmentations we want to apply to our data, we create a `Sequential` object containing them. We can specify or not if we want them to occurs at a random order with `random_order`.

```
seq = iaa.Sequential([ iaa.Dropout(p=0.2), iaa.Affine(rotate=(-30,30))], random_order= True)
```

#### Data augmentation on a batch during training

Augmentation can't be done when initializing the data class because we want the data to vary for each iteration, but also not when calling `__getitem__` because the augmentation would be done for each image called individually, creating a bottleneck. We want to process the data when fetching a whole batch.

It is a best practice to augment on top of a batch of images than doing so one image at a time, as it is considerably faster.
In our `Dataset` class, because we cannot use `__getitem__` which fetches one data point at a time, we'll use a function that we call `collate_fn`. This function checks if and `augmenters` sequential attribute has been set (**as we do not want to do augmentation during validation**, and given a batch, separates images and class before performing augmentation and recreating the tensor:

```
def collate_fn(self, batch): 
	ims, classes = list(zip(*batch))
	if self.aug: 
		ims=self.aug.augment_images(images=ims) 
		ims = torch.tensor(ims)[:,None,:,:].to(device)/255 
		classes = torch.tensor(classes).to(device) 
		return ims, classes
```

When calling the `DataLoader`, a new argument is gonna be specified to leverage this function of our `Dataset` class:

```
trn_dl = DataLoader(train, batch_size=64, \
    collate_fn=train.collate_fn,shuffle=True)
```

The training can now proceed as usual. Here we redo our translation test on a model with augmented data:
![[Pasted image 20230915161828.png]]

The model has learned to account for translation, data augmentation solved our initial problem.

