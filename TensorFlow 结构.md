## 一
We designed TensorFlow for large-scale distributed training and inference, but it is also flexible enough to support experimentation with new machine learning models and system-level optimizations. 

我们为大规模分布式培训和推理设计了TensorFlow，但它也足够灵活，可以支持新机器学习模型和系统级优化的实验。

This document describes the system architecture that makes this combination of scale and flexibility possible. It assumes that you have basic familiarity with TensorFlow programming concepts such as the computation graph, operations, and sessions. See this document for an introduction to these topics. Some familiarity with distributed TensorFlow will also be helpful.

本文档描述了可以实现规模和灵活性组合的系统架构。 它假定您基本熟悉TensorFlow编程概念，例如计算图，操作和会话。 有关这些主题的介绍，请参阅此文档。 熟悉分布式TensorFlow也很有帮助。

This document is for developers who want to extend TensorFlow in some way not supported by current APIs, hardware engineers who want to optimize for TensorFlow, implementers of machine learning systems working on scaling and distribution, or anyone who wants to look under Tensorflow's hood. By the end of this document you should understand the TensorFlow architecture well enough to read and modify the core TensorFlow code.

本文档适用于希望以当前API不支持的某种方式扩展TensorFlow的开发人员，希望针对TensorFlow进行优化的硬件工程师，负责扩展和分发的机器学习系统的实施者，或任何想要了解Tensorflow的人员。 在本文档的最后，您应该了解TensorFlow架构，以便读取和修改核心TensorFlow代码。

## Overview
TensorFlow运行时是一个跨平台库。图1说明了它的一般架构。 C API将不同语言的用户级代码与核心运行时分开。

![avatar](https://www.tensorflow.org/images/layers.png)

This document focuses on the following layers:

Client:
* Defines the computation as a dataflow graph. 将计算定义为数据流图。
* Initiates graph execution using a session. 使用会话启动图执行。

Distributed Master
* Prunes a specific subgraph from the graph, as defined by the arguments to Session.run(). 
修剪图中的特定子图，由Session.run（）的参数定义。
* Partitions the subgraph into multiple pieces that run in different processes and devices.
将子图分成多个部分，这些部分在不同的进程和设备中运行。
* Distributes the graph pieces to worker services.将图形片段分发给工作服务。
* Initiates graph piece execution by worker services.
通过工作服务启动图形块执行。

Worker Services (one for each task)
* Schedule the execution of graph operations using kernel implementations appropriate to the available hardware (CPUs, GPUs, etc).使用适合于可用硬件（CPU，GPU等）的内核实现来安排图形操作的执行。
* Send and receive operation results to and from other worker services.向其他工作服务发送和接收操作结果。

Kernel Implementations
* Perform the computation for individual graph operations.执行单个图形操作的计算。
