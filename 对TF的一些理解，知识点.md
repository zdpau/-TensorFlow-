## 1，有关placeholder与feed_dict的理解（https://blog.csdn.net/m0_37324740/article/details/77803694）
我的理解是，先用ph占住内存，然后当使用这个变量时，通过feed_dict来传递数据。
tf.placeholder(dtype, shape=None, name=None) 此函数可以理解为形参，用于定义过程，在执行的时候再赋具体的值,返回Tensor 类型。
## 2, 有关tf.Variable和tf.get_variable的区别（https://blog.csdn.net/u012436149/article/details/53696970  ）
tf.Variable() 每次都在创建新对象，所有reuse=True 和它并没有什么关系。对于get_variable()来说，如果已经创建的变量对象，就把那个对象返回，如果没有创建变量对象的话，就创建一个新的。

>* 使用tf.Variable时，如果检测到命名冲突，系统会自己处理。使用tf.get_variable()时，系统不会处理冲突，而会报错；
>* 当我们需要共享变量的时候，需要使用tf.get_variable()。在其他情况下，这两个的用法是一样的。
## 3，tf.variable_scope和tf.name_scope的用法（https://blog.csdn.net/uestc_c2_403/article/details/72328815 https://www.imooc.com/article/22966）
>* tf.variable_scope可以让变量有相同的命名，包括tf.get_variable得到的变量，还有tf.Variable的变量
>* tf.name_scope可以让变量有相同的命名，只是限于tf.Variable的变量。

name_scope 作用于操作，variable_scope 可以通过设置reuse 标志以及初始化方式来影响域下的变量。

**TF中有两种作用域类型**
命名域 (name scope)，通过tf.name_scope 或 tf.op_scope创建；
变量域 (variable scope)，通过tf.variable_scope 或 tf.variable_op_scope创建；
这两种作用域，对于使用tf.Variable()方式创建的变量，具有相同的效果，都会在变量名称前面，加上域名称。
对于通过tf.get_variable()方式创建的变量，只有variable scope名称会加到变量名称前面，而name scope不会作为前缀。例如 print(v1.name) # var1:0
**小结**：name_scope不会作为tf.get_variable变量的前缀，但是会作为tf.Variable的前缀。

## 4. 有关tf.shape, tensor.get_shape的用法（https://blog.csdn.net/jinlong_xu/article/details/71405305）
首先两种方法都可以获得变量的shape，不同点在于：

1，get_shape()不需要放在Session中即可运行；tf.shape不行。
2，只有tensor有这个方法tensor.get_shape()， 返回是一个tuple；tf.shape(x)其中x可以是tensor，也可不是tensor，类型可以是tensor, list, array，返回是一个tensor.
```
import tensorflow as tf
import numpy as np

x = tf.constant([[1,2,3],[4,5,6]])
y = [[1,2,3],[4,5,6]]
z = np.arange(24).reshape(2,3,4)

sess=tf.Session()
x_shape = tf.shape(x)
y_shape = tf.shape(y)
z_shape = tf.shape(z)
print sess.run(x_shape) # output:[2,3]
print sess.run(y_shape) # output:[2,3]
print sess.run(z_shape) # output:[2,3,4]
```
```
import tensorflow as tf
import numpy as np

x = tf.constant([[1,2,3],[4,5,6]])
y = [[1,2,3],[4,5,6]]
z = np.arange(24).reshape(2,3,4)

x_shape = x.get_shape() # 输出（2，3），注意是圆括号。还要注意是x.get_shape(这里没参数)
z_shape = z.get_shape() #这种就出错了，因为只能是Tensor数据类型才能使用这个方法。
```
**我们经常会这样来feed数据,如果在运行的时候想知道None到底是多少,这时候,只能通过tf.shape(x)[0]这种方式来获得.**


## 5，有关tf.argmax()以及axis解析的用法（https://blog.csdn.net/qq575379110/article/details/70538051/ https://blog.csdn.net/u012436149/article/details/52905166）
简单地说tf.argmax就是返回最大的那个数值所在的下标。 

## 6,tensor和variable的区别（https://blog.csdn.net/silent56_th/article/details/75577974）
```
a = tf.Variable(tf.ones(1))
b = tf.add(a,tf.ones(1))
```
a是Variable，而b是Tensor。

>Variable是可更改的（mutable），而Tensor是不可更改的。一个直接的例子就是Tensor不具有assign函数，而Variable含有。

>python和其他语言的API以及实现方式存在差异，本文只探讨general以及python方面的内容。

>Variable用于存储网络中的权重矩阵等变量，而Tensor更多的是中间结果等。

>Variable是会显示分配内存空间的（既可以是内存，也可以是显存），需要初始化操作（assign一个tensor），由Session管理，可以进行存储、读取、更改等操作。相反地，诸如Const, Zeros等操作创造的Tensor，是记录在Graph中，所以没有单独的内存空间；而其他未知的由其他Tensor操作得来的Tensor则是只会在程序运行中间出现。

>Tensor可以使用的地方，几乎都可以使用Variable。
## 7,有关tf.app.flags的知识点(https://blog.csdn.net/lyc_yongcai/article/details/73456960)
```
tf.app.flags.DEFINE_string("param_name", "default_val", "description")  
# 如果只写 python app_flags.py 则代码运行时默认程序里面设置的默认设置  

# 若 python app_flags.py --train_data_path <绝对路径 train.txt> --max_sentence_len 100  
#    --embedding_size 100 --learning_rate 0.05  代码再执行的时候将会按照上面的参数来运行程序  
```
```
# 使用这种方式保证了，如果此文件被其他文件 import的时候，不会执行main 函数  
if __name__ == '__main__':  
    tf.app.run()   # 解析命令行参数，调用main 函数 main(sys.argv)
```
