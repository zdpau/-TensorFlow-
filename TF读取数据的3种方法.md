https://blog.csdn.net/margretwg/article/details/70168256（cifar_input例子讲解，很详细）

https://blog.csdn.net/u012759136/article/details/52232266 （跟书上一样但相对全一些，但是后面的注意事项没有，以后用到再说）

https://blog.csdn.net/u010329292/article/details/68484485（这个有图，容易理解）

### Tensorflow 程序读取数据一共有3种方法：
>* 供给数据（feeding）：在程序运行的每一步，让Python代码来供给数据
>* 从文件读取数据： 让一个输入管线从文件中读取数据
>* 预加载数据：在tensorflow图中定义常量或变量来保存所有数据（适用于数据量小的时候）


一个典型的文件读取管线会包含下面这些步骤：
>* 1,文件名列表
>* 2,可配置的 文件名乱序(shuffling)
>* 3,可配置的 最大训练迭代数(epoch limit)
>* 4,文件名队列
>* 5,针对输入文件格式的阅读器
>* 6,纪录解析器
>* 7,可配置的预处理器
>* 8,样本队列
