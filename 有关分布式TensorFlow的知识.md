## 一， 对官方文档的翻译与理解
### 1， The tf.train.Server.create_local_server() method 使用进程内服务器创建单进程群集。(creates a single-process cluster, with an in-process server).

### 2， create cluster
#### 1， TensorFlow“集群”是一组参与TensorFlow图形分布式执行的“任务”。每个任务都与一个TensorFlow“服务器”相关联，该服务器包含一个可用于创建会话的“master”和一个在图中执行操作的“worker”。一个集群也可以分成一个或多个“作业”，每个作业包含一个或多个任务。(A TensorFlow "cluster" is a set of "tasks" that participate in the distributed execution of a TensorFlow graph. Each task is associated with a TensorFlow "server", which contains a "master" that can be used to create sessions, and a "worker" that executes operations in the graph. A cluster can also be divided into one or more "jobs", where each job contains one or more tasks.)

#### 2, 要创建集群，您需要为集群中的每个任务启动一台TensorFlow服务器。每个任务通常运行在不同的机器上，但您可以在同一台机器上运行多个任务（例如，以控制不同的GPU设备）。在每个任务下，你可以做：① 创建一个描述集群中所有任务的tf.train.ClusterSpec。这对每个任务都应该是一样的。（reate a tf.train.ClusterSpec that describes all of the tasks in the cluster. This should be the same for each task.）② 创建一个tf.train.Server，将tf.train.ClusterSpec传递给构造函数，并使用作业名称和任务索引标识本地任务。（Create a tf.train.Server, passing the tf.train.ClusterSpec to the constructor, and identifying the local task with a job name and task index.）

#### 3,  创建一个tf.train.ClusterSpec来描述集群:

##### 1,



参考文献：https://www.tensorflow.org/deploy/distributed
