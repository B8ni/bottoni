---
toc: true
layout: post
description: Wikibot Neural Network.
categories: [wiki, neural, network]
title: Neural Network
---

## One Liner Definition
Neural Network is a mathematical function. 

This mathematical function is based on inputs and parameters.

Inputs and parameters are multiplied and added them up. Negative values are set to zero.

These operations are repeated untill the error of prediction is minimized.

## Motivation

These 3 simple steps are the foundation of any deep learning model.

Implicitally they touch most important parts of NN:
1. Inputs and parameters are multiplied and added them up => [Matrix multiplication]();
2. Negative values are set to zero => [Rectified Linear function]();
3. Operations repeated untill error of prediction is minimized => [Gradient descent]() on [Loss function]().

The most complex deep learning model is built on these foundamentals. Deeply understanding them will help to breaking every complex model out.

## Implementation


```python
def f(x): return  3*x**2 + 2*x + 1
```


```python
def quad(a, b, c, x): return a*x**2 + b*x + c
```


```python
def noise(x, scale): return np.random.normal(scale=scale, size=x.shape)
def add_noise(x, mult, add): return x * (1+noise(x,mult)) + noise(x,add)
```


```python
import numpy as np
import torch

np.random.seed(42)

x = torch.linspace(-2, 2, steps=20)[:,None]
y = add_noise(f(x), 0.15, 1.5)
```


```python
from functools import partial

def mk_quad(a, b, c): return partial(quad, a, b, c)
```


```python
def mae(pred, actual): return torch.abs(pred-actual).mean()
```


```python
def quad_mae(params):
    f = mk_quad(*params)
    return mae(f(x), y)
```


```python
import torch
abc = torch.Tensor([1.1, 1.1, 1.1])
```


```python
abc.requires_grad_()

loss = quad_mae(abc)
```


```python
for i in range(10):
    loss = quad_mae(abc)
    loss.backward()
    with torch.no_grad(): abc -= abc.grad*0.01
    print(f'step={i}; loss={loss:.2f}')
```

    step=0; loss=2.42
    step=1; loss=2.40
    step=2; loss=2.36
    step=3; loss=2.30
    step=4; loss=2.21
    step=5; loss=2.11
    step=6; loss=1.98
    step=7; loss=1.85
    step=8; loss=1.72
    step=9; loss=1.58


## Take Away

1. [Matrix multiplication]() is the key to quickly calculate multuplication and addition of inputs and parameters.
2. [Gradient descent]() is the tool used to understand how to minimize the loss function, since loss function composed by parameters ``abc``.
3. [Rectified Linear function](), known as [ReLU](), is a linear function which takes input and the ouput is equals to the input. If the input is negative the output is zero. It's defined as: $f(x) = max(0, x)$

### Further Work
1. Ankify:
    - [ ] Matrix multiplication
    - [ ] Gradient descent
    - [ ] ReLU
    - [ ] End to end GD which every DL model is based on
2. [ ] Develop a Neural Network from scratch

---
## References
- [Neural net foundations - Jeremy Howard, 2022](https://course.fast.ai/Lessons/lesson3.html)
- [How does a neural net really work? - Jeremy Howard, 2022](https://www.kaggle.com/code/jhoward/how-does-a-neural-net-really-work)
- Chatting with [ChatGPT](https://chat.openai.com/)
