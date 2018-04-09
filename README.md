# -TensorFlow-
第一本书：Getting Started with TensorFlow

source tensorflow/bin/activate   

(source) deactivate

使用的是virtualenv来运行TensorFlow

1,安装virtualenv：
sudo easy_install pip  # If pip is not already installed

sudo pip install --upgrade virtualenv

2，Create a virtualenv environment by issuing a command of one of the following formats:

 $ virtualenv --system-site-packages targetDirectory # for Python 2.7
 
 $ virtualenv --system-site-packages -p python3 targetDirectory # for Python 3.n
 
 where targetDirectory identifies the top of the virtualenv tree. Our instructions assume that targetDirectory is ~/tensorflow, but you may choose any directory.
 
3，Activate the virtualenv environment by issuing one of the following commands:

$ source ~/tensorflow/bin/activate      # If using bash, sh, ksh, or zsh

$ source ~/tensorflow/bin/activate.csh  # If using csh or tcsh 

4，退出:(source) deactivate 

## tf.app.flags
tf.app.flags is a wrapper for the Python argparse module, which is commonly used to process command-line arguments, with some extra and specific functionality.  tf.app.flags是一个Python argparse模块封装，这是常用的处理命令行参数，与一些额外的特定功能。

for instance, a Python command-line program with typical command-line arguments:

   `python distribute.py --job_name="ps" --task_index=0`

The program distribute.py is passed the following:  job_name="ps"   task_index=0

This information is then extracted within the Python script, by using:

   `tf.app.flags.DEFINE_string("job_name", "", "name of job")`
   
   `tf.app.flags.DEFINE_integer("task_index", 0, "Index of task")`

tf.app.flags.flags是一个结构，包含所有参数解析从命令行输入的值。

