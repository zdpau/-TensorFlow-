## 一， 有关数据并行，模型并行的讲解：
### https://www.zhihu.com/question/53851014
### https://blog.csdn.net/xsc_c/article/details/42420167
#### 介绍一种卷积神经网络训练过程中的SGD的并行化方法。
#### 两个变种：
##### 模型并行： 不同的 workers 训练模型的不同 patrs，比较适合神经元活动比较丰富的计算。
##### 数据并行： 不同的 workers 训练不同的数据案例，比较适合 weight 矩阵比较多的计算。
#### 现代卷积神经网络主要由两种层构成，他们具有不一样的属性和性能：
##### 卷积层，占据了90% ~ 95% 的计算量，5% 的参数，但是对结果具有很大的表达能力。
##### 全连接层，占据了 5% ~ 10% 的计算量， 95% 的参数，但是对于结果具有相对较小的表达的能力。
#### 综上：卷积层计算量大，所需参数系数 W 少，全连接层计算量小，所需参数系数 W 多。因此对于卷积层适合使用数据并行，对于全连接层适合使用模型并行。

## 二，分布式TensorFlow的理解：
### https://zhuanlan.zhihu.com/p/35083779 分布式TensorFlow入门教程）这篇比较重要，教的比较详细，有关数据模型并行，ASGD，SGD，stale gradient都讲了
### https://www.oreilly.com/ideas/distributed-tensorflow （这篇基本包含上面那一篇，不过是英文版，比较重要）
### https://blog.csdn.net/u012436149/article/details/53140869 （这篇讲得比较详细，需要再看）
### https://blog.csdn.net/luodongri/article/details/52596780 (讲的听通俗易懂的)
### https://www.jianshu.com/p/bf17ac9e6357 （简书，可以借鉴一下）
### https://www.tensorflow.org/extend/architecture#code （当初PPT内容）
### https://www.tensorflow.org/deploy/distributed

## 三、查看机器上GPU情况

命令： nvidia-smi

功能：显示机器上gpu的情况

命令： nvidia-smi -l

功能：定时更新显示机器上gpu的情况



## 四，TensorFlow指定特定GPU以及占用显存的比例（使用某一块GPU, 多块, 禁用）
### https://blog.csdn.net/m0_37041325/article/details/77488981 

### https://blog.csdn.net/u014381600/article/details/72911262

### 1，可设置环境变量CUDA_VISIBLE_DEVICES，指明可见的cuda设备

方法1： 在/etc/profile或~/.bashrc的配置文件中配置环境变量(/etc/profile影响所有用户，~/.bashrc影响当前用户使用的bash shell)

在/etc/profile文件末尾添加以下行：

export CUDA_VISIBLE_DEVICES=0,1 ##仅显卡设备0,1GPU可见。可用的GPU可通过nvidia-smi -L命令查看

:wq保存并退出

source /etc/profile使配置文件生效

方法2：若上述配置无效，或者，如果用方法1，虽然方便，但有的时候还是需要指定其他的GPU，这时可以可在执行cuda程序时指明参数，在终端设置使用的GPU,如
```
CUDA_VISIBLE_DEVICES=1 python my_script.py  #只会使用序号为1的GPU


Environment Variable Syntax      Results

CUDA_VISIBLE_DEVICES=1           Only device 1 will be seen
CUDA_VISIBLE_DEVICES=0,1         Devices 0 and 1 will be visible
CUDA_VISIBLE_DEVICES="0,1"       Same as above, quotation marks are optional
CUDA_VISIBLE_DEVICES=0,2,3       Devices 0, 2, 3 will be visible; device 1 is masked
CUDA_VISIBLE_DEVICES=""          No GPU will be visible
```

方法3.在程序中指定使用的GPU
import os

os.environ["CUDA_VISIBLE_DEVICES"]=‘6‘’，‘7’  这里就设置了使用序号为6,7两个的GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"    禁用GPU

### 2，设置tensorflow使用的显存大小（设置定量的GPU使用量）
默认tensorflow是使用GPU尽可能多的显存。可以通过下面的方式，来设置使用的GPU显存：
```
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))  
```
上面分配给tensorflow的GPU显存大小为：GPU实际显存*0.7。

可以按照需要，设置不同的值，来分配显存。

设置最小的GPU使用量,可能API已经变了
```
config = tf.ConfigProto() 
config.gpu_options.allow_growth = True 
session = tf.Session(config=config)
```
