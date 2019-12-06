# jazz
a home-made neural network framework

## 中文简介
这是我基于自己的python理解和对深度学习及其框架的理解，自制的深度学习框架。但由于缺乏cuda编程能力和对底层通信的知识了解，目前我只实现了部分核心，包括一些基础算子和基础优化器，一个可以在mnist上进行训练和测试的demo。数据接口只是简单封装了``numpy.array``，整体接口和风格模仿了PyTorch，部分借鉴了Tensorflow。关于这个框架的构建思路，一部分内容我写成了markdown放在docs文件夹中。后续会继续实现框架的其他功能，包括但不限于
- 自制数据类型及其CUDA实现
- 多线程数据加载器
- 更多的基础算子
- 网络可视化

## Introduction
This is a home-made deep learning framework that I build based on my understanding of the python language and the deep learning, which utilize many built-in functions and features of python. Due to the lack of the ability of CUDA programming and the knowledge of the underlying communication of computers, for now I am only able to fullfil part of the core of the framework, including some basic operators and optimizers, and one demo to train and test on mnist dataset (which of course does not reach the state-of-the-art performance, either in metrics or computationally). The basic tensor is only a wrapper of the ``numpy.array``. The overall interfaces are similar to PyTorch, and some to Tensorflow. The markdown files in the docs folder have recorded most of my thoughts on building such a framework, but currently only in Chinese.
More features are to be implemented including but not limited to
- A totally homemade basic data type and its CUDA implementation
- Multiprocessing dataloader
- More basic operators
- Network visualization

p.s. It's named Jazz, simply because I am recently so much into Jazz music.
