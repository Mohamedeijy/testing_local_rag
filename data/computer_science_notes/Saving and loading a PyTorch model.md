#### Reference: [[Modern Computer Vision with PyTorch]]
#### Subject: [[PyTorch]]
#### Tags: #model #pytorch

The important component that completely define a neural network are:
- a unique key/name for each tensor (parameter) **`__init__` phase usually takes care of this**
- the logic to connect every tensor in the network with one another **`forward` phase usually takes care of this**
- the values (weight/bias values) of each tensor 

## The model's state dict

`model.state_dict()` gives access to a dictionary that corresponds to the parameter names (keys) and the weight and bias values attached.

```
state refers to the current snapshot of the model (where the snapshot is the set of values at each tensor).
```

![[Pasted image 20230905165747.png]]

## Saving

```
Running torch.save(model.state_dict(), 'mymodel.pth') will save this model in a Python serialized format on the disk with the name mymodel.pth. A good practice is to transfer the model to the CPU before calling torch.save as this will save tensors as CPU tensors and not as CUDA tensors. This will help in loading the model onto any machine, whether it contains CUDA capabilities or not.
```

## Loading

`Loading a model would require us to initialize the model with random weights first and then load the weights from state_dict:`
- create an empty model the same way the targeted model was created when training
- load the model from disk and unserialize it to get a state dict of type `OrderedDict` by using 
	`state_dict = torch.load('mymodel.pth')`
- load the state dict onto our model with 
	`model.load_state_dict(state_dict)`
	register to device and make a prediction.

`If all the weight names are present in the model, then you would get a message saying all the keys were matched.`
`# <All keys matched successfully>`
