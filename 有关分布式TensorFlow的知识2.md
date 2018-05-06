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
