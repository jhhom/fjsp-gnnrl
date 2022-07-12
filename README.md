## Learning to dispatch for Flexible Job Shop Scheduling via Deep Reinforcement Learning
This project adapts method proposed in [this paper](https://arxiv.org/abs/2010.12367) to solve Flexible Job Shop.

### Installation

**Anaconda**

Create environment
```
conda env create -f conda_environment.yml
```

Then, activate environment
```
conda activate fjsp-rl
```

### Reproduce experimental results
Run
```
python test_model.py
```

For problem with static and stochastic job arrival times, run
```
python test_model_stochastic.py
```

You can use Jupyter notebook to visualize the resulting schedule:
```
test_model.ipynb
test_model_stochastic.ipynb
```

### Train new model
1. Go to `params.py`, change

```python
config.size = 'MK01'        # this is the problem size, either MK01, MK02, ... MK10
config.device = 'cuda'      # device to train on, 'cpu' or 'cuda'
config.stochastic = False   # set this to True if you would like to train on problems with stochastic arrival time attributed to each job
config.progress_config.path_to_save_progress = ''   # path to save model weights, training and validation logs, directory of this path must be empty
```

2. Run

```
python train.py
```


### Benchmark dataset
Dataset used for benchmark is available for download at:

https://people.idsia.ch/~monaldo/fjsp.html#Applications

### Paper reference
```
@inproceedings{NEURIPS2020_11958dfe,
 author = {Zhang, Cong and Song, Wen and Cao, Zhiguang and Zhang, Jie and Tan, Puay Siew and Chi, Xu},
 booktitle = {Advances in Neural Information Processing Systems},
 editor = {H. Larochelle and M. Ranzato and R. Hadsell and M. F. Balcan and H. Lin},
 pages = {1621--1632},
 publisher = {Curran Associates, Inc.},
 title = {Learning to Dispatch for Job Shop Scheduling via Deep Reinforcement Learning},
 url = {https://proceedings.neurips.cc/paper/2020/file/11958dfee29b6709f48a9ba0387a2431-Paper.pdf},
 volume = {33},
 year = {2020}
}
```
