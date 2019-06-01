# create common network structure (tensorflow)
我们在networks.py文件中定义常用的神经网络结构，他包含：Vgg16, Vgg19, ResNet18, ResNet34, 
ResNet50, ResNet101, ResNet152 

## How to use
### 1. find the model.py
举例：假设我们像定义ResNet152网络
````ruby
m4_ResNet152 = ResNet152(cfg)
m4_ResNet152.build_model(images)
````
在model.py文件中我们定义了一个ResNet152网络
````ruby
python model.py --log_dir='你存储tensorboard文件的路径'
````
会在指定的路径下生成一个以***系统当前时间命名的文件夹***

## 2. 网络结构：Vgg，ResNet
### 2.1 Vgg16  
#### 2.1.1 Vgg16类位于[networks.py](https://github.com/blueskyM01/Common-network-description/blob/master/networks.py)文件中的````Vgg16````， 定义的网络结构如下图所示：  
![image](https://github.com/blueskyM01/Common-network-description/blob/master/vgg.png)  
#### 2.1.2 卷积层(````conv_layer````)  
![image](https://github.com/blueskyM01/Common-network-description/blob/master/conv_layer.png)  
 函数位于[ops.py](https://github.com/blueskyM01/Common-network-description/blob/master/ops.py)中的````m4_conv_layers````
#### 2.1.3 卷积模块  
***Vgg中没有专门设计卷积模块， 直接按照[层次结构表](https://github.com/blueskyM01/Common-network-description/blob/master/pictures_of_network_structure/vgg.png)用卷积层(````conv_layer````)依次搭建的***  
#### 2.1.3 注意
***在送入softmax之前的那层没有激活函数***
### 2.2 Vgg19  
Vgg19与Vgg16大体相同，只是层次有所增加，见[层次结构表](https://github.com/blueskyM01/Common-network-description/blob/master/pictures_of_network_structure/vgg.png)中的E列






