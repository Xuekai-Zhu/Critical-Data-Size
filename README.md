# Critical Data Size of Language Models from a Grokking Perspective

This repo inculdes the offical code in the paper [Critical Data Size of Language Models from a Grokking Perspective](https://arxiv.org/pdf/2401.10463.pdf).

![Main_figure](figures/figure-1.png)

## Prerequisites

- `torch` >= 2.0
- `transformers`


## Quick Start

### 1. Grokking On IMDB

Execute the following command to re-produce our results: 

```shell

sh run_grokking_on_imdb.sh

```


### 2. Grokking on Yelp 


```shell
sh run_grokking_on_yelp.sh
```

## Citation
```shell
@article{zhu2024critical,
  title={Critical data size of language models from a grokking perspective},
  author={Zhu, Xuekai and Fu, Yao and Zhou, Bowen and Lin, Zhouhan},
  journal={arXiv preprint arXiv:2401.10463},
  year={2024}
}
```
