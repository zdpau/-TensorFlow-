## 1，有关placeholder与feed_dict的理解（https://blog.csdn.net/m0_37324740/article/details/77803694）
我的理解是，先用ph占住内存，然后当使用这个变量时，通过feed_dict来传递数据。
## 2, 有关tf.Variable和tf.get_variable的区别（https://blog.csdn.net/u012436149/article/details/53696970  ）
tf.Variable() 每次都在创建新对象，所有reuse=True 和它并没有什么关系。对于get_variable()来说，如果已经创建的变量对象，就把那个对象返回，如果没有创建变量对象的话，就创建一个新的。

>使用tf.Variable时，如果检测到命名冲突，系统会自己处理。使用tf.get_variable()时，系统不会处理冲突，而会报错；
>当我们需要共享变量的时候，需要使用tf.get_variable()。在其他情况下，这两个的用法是一样的。
## 3，tf.variable_scope和tf.name_scope的用法（https://blog.csdn.net/uestc_c2_403/article/details/72328815）
> tf.variable_scope可以让变量有相同的命名，包括tf.get_variable得到的变量，还有tf.Variable的变量
> tf.name_scope可以让变量有相同的命名，只是限于tf.Variable的变量。
