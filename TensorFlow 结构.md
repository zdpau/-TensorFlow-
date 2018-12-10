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

![avatar](https://www.tensorflow.org/images/diag1.svg)

Figure 2 illustrates the interaction of these components. "/job:worker/task:0" and "/job:ps/task:0" are both tasks with worker services. "PS" stands for "parameter server": a task responsible for storing and updating the model's parameters. Other tasks send updates to these parameters as they work on optimizing the parameters. This particular division of labor between tasks is not required, but is common for distributed training.

图2说明了这些组件的交互。 “/ job：worker / task：0”和“/ job：ps / task：0”都是具有工作服务的任务。 “PS”代表“参数服务器”：负责存储和更新模型参数的任务。 其他任务在优化参数时会对这些参数发送更新。 任务之间的这种特殊分工不是必需的，但对于分布式培训来说是常见的。

Note that the Distributed Master and Worker Service only exist in distributed TensorFlow. The single-process version of TensorFlow includes a special Session implementation that does everything the distributed master does but only communicates with devices in the local process.

The following sections describe the core TensorFlow layers in greater detail and step through the processing of an example graph.

请注意，Distributed Master和Worker Service仅存在于分布式TensorFlow中。 TensorFlow的单进程版本包含一个特殊的Session实现，它可以执行分布式主服务器执行的所有操作，但只与本地进程中的设备进行通信。

以下部分更详细地描述了核心TensorFlow层，并逐步处理示例图。

## Client
Users write the client TensorFlow program that builds the computation graph. This program can either directly compose individual operations or use a convenience library like the Estimators API to compose neural network layers and other higher-level abstractions. TensorFlow supports multiple client languages, and we have prioritized Python and C++, because our internal users are most familiar with these languages. As features become more established, we typically port them to C++, so that users can access an optimized implementation from all client languages. Most of the training libraries are still Python-only, but C++ does have support for efficient inference.

用户编写构建计算图的客户端TensorFlow程序。该程序可以直接组成单独的操作，也可以使用Estimators API之类的便利库来组合神经网络层和其他更高级别的抽象。 TensorFlow支持多种客户端语言，我们优先考虑Python和C ++，因为我们的内部用户最熟悉这些语言。随着功能变得更加成熟，我们通常将它们移植到C ++，以便用户可以从所有客户端语言访问优化的实现。大多数训练库仍然只支持Python，但C ++确实支持有效的推理。

The client creates a session, which sends the graph definition to the distributed master as a tf.GraphDef protocol buffer. When the client evaluates a node or nodes in the graph, the evaluation triggers a call to the distributed master to initiate computation.

In Figure 3, the client has built a graph that applies weights (w) to a feature vector (x), adds a bias term (b) and saves the result in a variable (s).

客户端创建会话，该会话将图形定义作为tf.GraphDef协议缓冲区发送到分布式主节点。当客户端评估图中的一个或多个节点时，评估会触发对分布式主节点的调用以启动计算。

在图3中，客户端构建了一个图表，将权重（w）应用于特征向量（x），添加偏差项（b）并将结果保存在变量中。

![avatar](https://www.tensorflow.org/images/graph_client.svg)

## Distributed master
The distributed master:

* prunes the graph to obtain the subgraph required to evaluate the nodes requested by the client, 修剪图形以获得评估客户端请求的节点所需的子图，
* partitions the graph to obtain graph pieces for each participating device, and 对图表进行分区以获取每个参与设备的图形片段，以及
* caches these pieces so that they may be re-used in subsequent steps.缓存这些碎片，以便它们可以在后续步骤中重复使用。

Since the master sees the overall computation for a step, it applies standard optimizations such as common subexpression elimination and constant folding. It then coordinates execution of the optimized subgraphs across a set of tasks.

由于主控器看到了步骤的整体计算，因此它应用标准优化，例如公共子表达式消除和常量折叠。 然后，它协调一组任务中优化子图的执行。

![avatar](https://www.tensorflow.org/images/graph_master_cln.svg)

Figure 5 shows a possible partition of our example graph. The distributed master has grouped the model parameters in order to place them together on the parameter server.图5(下面的)显示了示例图的可能分区。分布式主服务器已对模型参数进行分组，以便将它们放在参数服务器上。

![avatar](https://www.tensorflow.org/images/graph_split1.svg)

Where graph edges are cut by the partition, the distributed master inserts send and receive nodes to pass information between the distributed tasks (Figure 6).在分区切割图形边缘的情况下，分布式主控插入发送和接收节点以在分布式任务之间传递信息（图6）。

![avatar](https://www.tensorflow.org/images/graph_split2.svg)

The distributed master then ships the graph pieces to the distributed tasks.
然后，分布式主服务器将图形片段传送到分布式任务。

![avatar](https://www.tensorflow.org/images/graph_workers_cln.svg)

## Worker Service
The worker service in each task:每项任务中的工作服务：
* handles requests from the master,处理来自主人的请求，
* schedules the execution of the kernels for the operations that comprise a local subgraph, and 为包含本地子图的操作安排内核的执行，和
* mediates direct communication between tasks.调解任务之间的直接沟通。

We optimize the worker service for running large graphs with low overhead. Our current implementation can execute tens of thousands of subgraphs per second, which enables a large number of replicas to make rapid, fine-grained training steps. The worker service dispatches kernels to local devices and runs kernels in parallel when possible, for example by using multiple CPU cores or GPU streams.

我们优化工作服务，以便以较低的开销运行大型图形。 我们当前的实现可以每秒执行数万个子图，这使得大量复制副本可以进行快速，细粒度的训练步骤。 工作服务将内核分派给本地设备并在可能的情况下并行运行内核，例如通过使用多个CPU内核或GPU流。

We specialize Send and Recv operations for each pair of source and destination device types:我们专门针对每对源设备和目标设备类型的Send和Recv操作：
* Transfers between local CPU and GPU devices use the cudaMemcpyAsync() API to overlap computation and data transfer.本地CPU和GPU设备之间的传输使用cudaMemcpyAsync（）API来重叠计算和数据传输。
* Transfers between two local GPUs use peer-to-peer DMA, to avoid an expensive copy via the host CPU.两个本地GPU之间的传输使用对等DMA，以避免通过主机CPU进行昂贵的复制。

For transfers between tasks, TensorFlow uses multiple protocols, including:对于任务之间的传输，TensorFlow使用多种协议，包括：
* gRPC over TCP.
* RDMA over Converged Ethernet.

We also have preliminary support for NVIDIA's NCCL library for multi-GPU communication, see: tf.contrib.nccl. 我们还初步支持NVIDIA用于多GPU通信的NCCL库，请参阅：tf.contrib.nccl。

![avatar](https://www.tensorflow.org/images/graph_send_recv.svg)

## Kernel Implementations
The runtime contains over 200 standard operations including mathematical, array manipulation, control flow, and state management operations. Each of these operations can have kernel implementations optimized for a variety of devices. Many of the operation kernels are implemented using Eigen::Tensor, which uses C++ templates to generate efficient parallel code for multicore CPUs and GPUs; however, we liberally use libraries like cuDNN where a more efficient kernel implementation is possible. We have also implemented quantization, which enables faster inference in environments such as mobile devices and high-throughput datacenter applications, and use the gemmlowp low-precision matrix library to accelerate quantized computation.

运行时包含200多个标准操作，包括数学，数组操作，控制流和状态管理操作。这些操作中的每一个都可以具有针对各种设备优化的内核实现。许多操作内核都是使用Eigen :: Tensor实现的，它使用C ++模板为多核CPU和GPU生成高效的并行代码;但是，我们自由地使用像cuDNN这样的库，可以实现更高效的内核实现。我们还实现了量化，可以在移动设备和高吞吐量数据中心应用等环境中实现更快的推理，并使用gemmlowp低精度矩阵库来加速量化计算。

If it is difficult or inefficient to represent a subcomputation as a composition of operations, users can register additional kernels that provide an efficient implementation written in C++. For example, we recommend registering your own fused kernels for some performance critical operations, such as the ReLU and Sigmoid activation functions and their corresponding gradients. The XLA Compiler has an experimental implementation of automatic kernel fusion.

如果将子计算表示为操作组合是困难或低效的，则用户可以注册提供用C ++编写的有效实现的其他内核。例如，我们建议为一些性能关键操作注册自己的融合内核，例如ReLU和Sigmoid激活函数及其相应的渐变。 XLA编译器具有自动内核融合的实验性实现。
