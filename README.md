# pytorch-seq2seq
This is a repo of neural machine translate using seq2seq model, based on https://github.com/bentrevett/pytorch-seq2seq.
 Now it only supports LSTM. In the future we will add other transformers like BERT.

## Downloading Datasets
Create a directory data/ .
```
mkdir data
```

Download 2 files from https://drive.google.com/open?id=1yy-r2F_YrUxtPzDQMYJwqQ4IOgtT1z-k under data/.


## Experiments
By default we do not use any GPU. If you wanna use GPU(s), you need to specify GPU id(s). Note that multi-GPU training is supported here.
```
python train.py --gpu=0
python train.py --gpu=0,1
```

The original dataset is very huge. You can only specify the number of samples you want to use. By default we use all samples.
Note that more samples result in more trainable parameters and thus more CUDA memory.
```
python train.py --gpu=0 --n_samples=10000

```

Other hyper-parameters are in param_parser.py, which are all self-explained. You can tune the parameters accordingly.

Other tips to note:
1) The source language is input in the reversed order.
2) Our decoder loop starts at 1, not 0. This means the 0th element of our outputs tensor remains all zeros. 
So when we calculate the loss, we cut off the first element of each tensor.