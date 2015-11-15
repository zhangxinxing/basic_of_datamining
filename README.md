# basic_of_datamining
basic_of_datamining
利用python中的pandas，sklean进行数据挖掘

pandas用来对数据集进行处理
sklean中提供了一些机器学习方法的实现

要利用这两个库，首先应该安装，其中numpy，scipy是两个比较重要的依赖库

在Ubuntu下，pandas安装可以使用：
sudo apt-get install python-pandas


sklean的安装可以参照：
http://blog.csdn.net/wbgxx333/article/details/12168675




一个数据挖掘的整体流程，主要包括
1.      定义问题
2.      准备数据
3.      浏览数据
4.      生成模型
5.      浏览和验证模型
6.      部署和更新模型

而在实现时，通俗的讲为：

1 加载训练数据
2 对数据进行各种预处理（抽样，去噪等）
3 提取特征
4 训练模型
5 利用模型对预测数据进行预测
6 预测结果评价（评分）
不断调整2 3 4 5 6步，进行优化

下面的例子，实现了这样一个基本流程
提供 刚刚入门 数据挖掘 概念者

一个较为基础的python+pandas+sklearn实现

来自一个比赛：
http://tianchi.aliyun.com/competition/introduction.htm?spm=5176.100066.333.4.scNYuk&raceId=5

对问题进行了简化处理，并没有将微博的内容和时间考虑在内


是对其比较初步的处理


数据来源是 来自阿里巴巴天池平台和新浪微博

这个地址：http://tianchi.aliyun.com/datalab/index.htm?spm=5176.100067.1234.4.t8WkFa（以后会更新）

例子的地址在：


在这个地址中文件，有相应数据文件和源码，同时会有对有文件和源码的解释
源码中，因为是针对特定问题的，所以一些对数据的加载、处理 或 评分细节不必深究，
重要的是整体的流程
源码中保留了一些被注释了的代码，大多是一些调试代码

对应的测试大数据集合下载地址：

http://pan.baidu.com/s/1qW49CcO
http://download.csdn.net/detail/xinxing__8185/9270655
