# Imbalances in NeuroSymbolic Learning: Characterization and Mitigating Strategies

This is an implementation of the training and testing time imbalance mitigation techniques proposed at [NeurIPS 2025](https://openreview.net/pdf?id=nik6BjmLm2)


## Computational Infrastructure 

The experiments ran on an Ubuntu 22.04.3 LTS machine with 3.16TB hard disk and an
NVIDIA GeForce RTX 2080 Ti GPU with 11264 MiB RAM. Our source code was implemented in Python 3.9.

## Requirements

The file ```requirements.txt``` has all the requirements to create our virtual environment.  


## Datasets
- MNIST (Creative Commons Attribution-Share Alike 3.0)
- CIFAR-10: https://www.cs.toronto.edu/~kriz/cifar.html
- HWF: https://liqing.io/NGS/ 

The first two datasets are available via PyTorch, so the user does not need to download them in advance.  
The last has to be downloaded by the users.

## Folder Structure 
- train_scallop.py: Python code to run SL (with/without RECORDS), LP(EMP), LP(ALG1) for MAX-M, SUM-M, HWF-M
- train_smallest_parent.py: script to run SL (with/without RECORDS), LP(EMP), and LP(ALG1) for Smallest Parent
- testing_time: Python code to run LA and CAROT for an input model. Supports 
 -- imbalance-free training and testing, as well as, 
 -- long-tail training and long-tail testing.
- scripts: example linux shell scripts for running our experiments using SL, SL+RECORDS, LP(EMP), LP(ALG1).

## Training Arguments
- ```--size_partial_dataset```: number of partial training samples.
- ```--M```: number of instances per training sample, as in our paper.   
- ```--records```: if it is on, then SL runs jointly with RECORDS (see the experiments for further details). 
- ```--top-k```: number of top-k proofs to consider during training (for efficiency reasons, all our experiments on MAX-M, SUM-M, HWF-M ran with k=1).
- ```--ilp-training```: if it is on, then training is carried under LP.
- ```--epsilon_ilp```: the epsilon approximation in Eq. (6).
- ```--ilp_solver```: solver for linear programs. At the moment, pywraplp is supported from https://developers.google.com/optimization/install/python.
- ```--continuous-relaxation```: if set to true, then continuous relaxations are adopted to solve Eq. (6).
- ```--gamma```, ```--est_epochs```, and ```--estimated-label-ratio``` are defined as in SoLar for empirical distribution estimation.
- ```--estimated-label-ratio```: techique to compute the partial ratios. ```gold``` is for the ratios computed using the ground-truth hidden gold labels. ```partial``` is for using Algorithm 1 to compute the ratios, 
```naive``` is for using SoLar's sliding window-based approximate technique.   

Instructions to run the Python code under folder ```testing_time``` are given inside the files. 


## Thanks 

- We used code from https://github.com/MediaBrain-SJTU/RECORDS-LTPLL to implement RECORDS (MIT License).
- We used code from https://github.com/hbzju/SoLar to implement a sliding-window based technique for estimating the hidden label priors.


## Citing Our Work

If you want to use our code, please cite the following: 

```
@inproceedings{NeurIPS2025a,
  abbr={NeurIPS},
  author       = {Efthymia Tsamoura and Kaifu Wang and Dan Roth},
  title        = {Imbalances in Neurosymbolic Learning: Characterization and Mitigating Strategies},
  booktitle    = {Proceedings of the Thirty-Ninth Conference on Neural Information Processing Systems (NeurIPS)},
  year         = {2025}
}
``` 