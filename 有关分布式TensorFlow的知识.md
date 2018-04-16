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

参考文献：1，https://www.tensorflow.org/deploy/distributed
