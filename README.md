## Zero to GANs  Course Project

**WORK IN PROGRSS**

Links
* https://mschmidt3.github.io/zero_to_gans/  - this file
* https://www.kaggle.com/mschmidt3/zero-to-gans-project - the kaggle notebook
* [move-to-colab](move to colab)

Selected Dataset: 
* intel image classification

First steps try to get parameter for normalization.
The result was terrible in ever batch there were some completely black ans some completely white images.

## The Project.

I choose the intel-image-classification data set I found on Kaggle.
The goal of the project is to set up an resnet similar to the on we used in the course.

The dataset contains anotated images. 
The goal is to train an convolutional network based on this data.

## First steps

* use the course project 05b-cifar10-resnet as starter.
* add the dataset to the kaggle notebook.


## Lessons learned so far

### perpare the data

Not all Images in the dataset have the same dimensions. This must be fixed before training.

```
...
history = [evaluate(model, valid_dl)]
```


The call of evaluate results in

```
--------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
<ipython-input-19-1c4791c5f87e> in <module>
----> 1 history = [evaluate(model, valid_dl)]
      2 history

/opt/conda/lib/python3.7/site-packages/torch/autograd/grad_mode.py in decorate_context(*args, **kwargs)
     13         def decorate_context(*args, **kwargs):
     14             with self:
---> 15                 return func(*args, **kwargs)
     16         return decorate_context
     17 

...

RuntimeError: Caught RuntimeError in DataLoader worker process 1.
Original Traceback (most recent call last):
  File "/opt/conda/lib/python3.7/site-packages/torch/utils/data/_utils/worker.py", line 178, in _worker_loop
    data = fetcher.fetch(index)
  File "/opt/conda/lib/python3.7/site-packages/torch/utils/data/_utils/fetch.py", line 47, in fetch
    return self.collate_fn(data)
  File "/opt/conda/lib/python3.7/site-packages/torch/utils/data/_utils/collate.py", line 79, in default_collate
    return [default_collate(samples) for samples in transposed]
  File "/opt/conda/lib/python3.7/site-packages/torch/utils/data/_utils/collate.py", line 79, in <listcomp>
    return [default_collate(samples) for samples in transposed]
  File "/opt/conda/lib/python3.7/site-packages/torch/utils/data/_utils/collate.py", line 55, in default_collate
    return torch.stack(batch, 0, out=out)
RuntimeError: stack expects each tensor to be equal size, but got [3, 150, 150] at entry 0 and [3, 141, 150] at entry 241
```

Adding `tt.Resize(150)` to the Compose function did not fix the error.

```
stats = ((0.43531275, 0.46185786, 0.4556407), (0.26646963, 0.26392624, 0.29387024))
train_tfms = tt.Compose([tt.RandomCrop(32, padding=4, padding_mode='reflect'), 
                         tt.RandomHorizontalFlip(), 
                         tt.Resize(150),
                         tt.ToTensor(), 
                         tt.Normalize(*stats,inplace=True)])
```

Adding `tt.Resize( (150,150) )`  did. 
Take care: `tt.Resize( (150,150) )` must be added to `tran_tfmd` and to `valid_tmfs`


### get better results.

1. `tt.Normalize(*stats ...)` does not improve the result significantly
1. adding another liniar layer did not have the the expected effect
