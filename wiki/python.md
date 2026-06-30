## Pytorch

### Device assignation commands:

```python
my_tensor = torch.tensor([10])  # A tensor
print(torch.rand(10).device)  # Show device that tensor is stored on
torch.rand(10).is_cuda        # True if tensor is a GPU tensor
  
torch.tensor([10], device=device) # Set tensor to device directly
my_model = MyRNN().to(device) # Set model to device directly
  
my_tensor = my_tensor.to(device) # Convert tensor to device
my_model.to(device) # Convert model to device (in place) # Model Input has to be on same device!
  
all(p.is_cuda for p in my_model.parameters()) # Check if whole model is on GPU (could be used to find out which are not too, or to move whole model to correct device avter loading from file)
  
# Set and check tensor dtype
# https://pytorch.org/docs/stable/tensors.html#initializing-and-basic-operations

my_tensor = torch.tensor(100, dtype=torch.float)
my_tensor.dtype


### Other Commands:
tensor.item() # Get single value of tensor (scalar)
tensor.cpu().detach().numpy() # detach: array instead of tensor?
```

### Parallel Computing (use multiple GPUs)
https://discuss.pytorch.org/t/run-pytorch-on-multiple-gpus/20932/62

## Jupyter

```python
%timeit -n 10000 fun()
```