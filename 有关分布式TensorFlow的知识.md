## 一， 对官方文档的翻译与理解
### 1， The tf.train.Server.create_local_server() method 使用进程内服务器创建单进程群集。(creates a single-process cluster, with an in-process server).

### 2， create cluster
#### 1， TensorFlow“集群”是一组参与TensorFlow图形分布式执行的“任务”。每个任务都与一个TensorFlow“服务器”相关联，该服务器包含一个可用于创建会话的“master”和一个在图中执行操作的“worker”。一个集群也可以分成一个或多个“作业”，每个作业包含一个或多个任务。(A TensorFlow "cluster" is a set of "tasks" that participate in the distributed execution of a TensorFlow graph. Each task is associated with a TensorFlow "server", which contains a "master" that can be used to create sessions, and a "worker" that executes operations in the graph. A cluster can also be divided into one or more "jobs", where each job contains one or more tasks.)

#### 2, 要创建集群，您需要为集群中的每个任务启动一台TensorFlow服务器。每个任务通常运行在不同的机器上，但您可以在同一台机器上运行多个任务（例如，以控制不同的GPU设备）。在每个任务下，你可以做：① 创建一个描述集群中所有任务的tf.train.ClusterSpec。这对每个任务都应该是一样的。（reate a tf.train.ClusterSpec that describes all of the tasks in the cluster. This should be the same for each task.）② 创建一个tf.train.Server，将tf.train.ClusterSpec传递给构造函数，并使用作业名称和任务索引标识本地任务。（Create a tf.train.Server, passing the tf.train.ClusterSpec to the constructor, and identifying the local task with a job name and task index.）

#### 3,  创建一个tf.train.ClusterSpec来描述集群:
##### 1, 群集规范字典将作业名称映射到网络地址列表。将此字典传递给tf.train.ClusterSpec构造函数。(The cluster specification dictionary maps job names to lists of network addresses. Pass this dictionary to the tf.train.ClusterSpec constructor.) 代码见参考文献1.

#### 4， 在每个任务中创建一个tf.train.Server实例：
##### 1， tf.train.Server对象包含一组本地设备，一组到tf.train.ClusterSpec中的其他任务的连接，以及一个可用于执行分布式计算的tf.Session。每个服务器都是特定命名作业的成员，并且在该作业中有一个任务索引。服务器可以与群集中的任何其他服务器进行通信。(A tf.train.Server object contains a set of local devices, a set of connections to other tasks in its tf.train.ClusterSpec, and a tf.Session that can use these to perform a distributed computation. Each server is a member of a specific named job and has a task index within that job. A server can communicate with any other server in the cluster.) 代码见参考文献1.

### 3, 指定模型中的分布式设备
#### 1, 要将操作放置在特定进程上，可以使用相同的tf.device函数来指定ops是否在CPU或GPU上运行。 代码见参考文献1

### 4，Replicated training
#### 1，“数据并行性”(一种常用的训练配置)：worker job中的多个task，在不同的小批量数据上训练相同的模型，更新托管在ps job中的一个或多个任务中的共享参数。所有tasks通常运行在不同的机器上。在TensorFlow中有很多方法可以指定这个结构，我们正在构建库，以简化指定复制模型的工作。可能的方法包括：
##### 1, 图内复制（In-graph replication）：在这种方法中，客户端构建一个包含一组参数的tf.Graph（在tf.Variable节点中固定为/ job：ps）;以及模型的计算密集型部分的多个副本，每个副本都固定在/ job：worker中的不同任务。
##### 2，图间复制（Between-graph replication）：对于每一个/job:worker task都有一个独立的客户端，通常与worker task处于相同的进程中,每个client构建一个包含参数的类似图(固定到/ job：ps像以前一样使用tf.train.replica_device_setter将它们确定性地映射到相同的任务)以及模型的计算密集型部分的单个副本，固定到/ job：worker中的本地任务。
##### 3，异步训练：在这种方法中，图的每个副本都有独立的训练循环that无需协调即可执行。它与以上两种复制形式兼容。
##### 4，同步训练：在此方法中，所有副本都读取当前参数的相同值，并行计算梯度，然后将它们应用到一起。它与图内复制（例如使用CIFAR-10多GPU训练器中的梯度平均）和图间复制兼容（例如使用tf.train.SyncReplicasOptimizer）。

## 2，解读与理解
可以看出，所谓的TensorFlow集群就是一组任务，每个任务就是一个服务。服务由两个部分组成，第一部分是master，用于创建session，第二部分是worker，用于执行具体的计算。

TensorFlow一般将任务分为两类job：一类叫参数服务器，parameter server，简称为ps，用于存储tf.Variable；一类就是普通任务，称为worker，用于执行具体的计算。

首先来理解一下参数服务器的概念。一般而言，机器学习的参数训练过程可以划分为两个类别：第一个是根据参数算算梯度，第二个是根据梯度更新参数。对于小规模训练，数据量不大，参数数量不多，一个CPU就足够了，两类任务都交给一个CPU来做。对于普通的中等规模的训练，数据量比较大，参数数量不多，计算梯度的任务负荷较重，参数更新的任务负荷较轻，所以将第一类任务交给若干个CPU或GPU去做，第二类任务交给一个CPU即可。对于超大规模的训练，数据量大、参数多，不仅计算梯度的任务要部署到多个CPU或GPU上，而且更新参数的任务也要部署到多个CPU。如果计算量足够大，一台机器能搭载的CPU和GPU数量有限，就需要多台机器来进行计算能力的扩展了。参数服务器是一套分布式存储，用于保存参数，并提供参数更新的操作。

## 3, glossary(词汇表)
1，client:client通常是一个构建TensorFlow图并构造tensorflow :: Session以与集群进行交互的程序。client通常使用Python或C++编写。单个客户端进程可以直接与多台TensorFlow服务器交互（请参阅上面的“复制式培训”），并且单台服务器可以为多个客户端提供服务。

2, cluster: TensorFlow集群包含一个或多个“作业(jobs)”，每个“作业”分为一个或多个“任务(tasks)”列表。cluster通常专用于特定的高级目标(high-level objective)，例如训练神经网络，并行使用多台机器。cluster由tf.train.ClusterSpec对象定义。

3, job: 一份工作(job)包括一份“任务(task)”清单，通常用于共同目的。例如，名为ps的作业（用于“参数服务器”）通常承载存储和更新变量的节点;而名为Worker的作业通常承载执行计算密集型任务的无状态节点。作业中的任务(the tasks in a job)通常运行在不同的机器上。这组工作角色是灵活的：例如，worker可能会保持某种状态。

4, Master service: RPC service提供对一组分布式设备的远程访问，并充当会话目标。master service实现tensorflow :: Session接口，并负责协调跨一个或多个“worker services”的工作。所有TensorFlow服务器均实施主服务(All TensorFlow servers implement the master service)。

5, task:任务对应于特定的TensorFlow服务器，并且通常对应于单个进程。任务属于特定的“工作（job）”，并通过其在该工作任务列表中的索引来标识。

6，TensorFlow server: 运行tf.train.Server实例的进程，该实例是集群的成员，并导出“主服务”和“worer service”。

7, worker service: 使用本地设备执行TensorFlow图形部分的RPC服务。一个worker service实现worker_service.proto。所有的TensorFlow服务器都实现了工作服务。(All TensorFlow servers implement the worker service.)



参考文献：1，https://www.tensorflow.org/deploy/distributed
        2, https://segmentfault.com/a/1190000008376957?from=timeline&isappinstalled=1
