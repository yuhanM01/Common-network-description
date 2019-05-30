# create common network structure
我们在networks.py文件中定义常用的神经网络结构，他包含：Vgg16, Vgg19, ResNet18, ResNet34, 
ResNet50, ResNet101, ResNet152 

## How to use
### 1. find the model.py
举例：假设我们像定义ResNet152网络  
'''m4_ResNet152 = ResNet152(cfg)  
m4_ResNet152.build_model(images)'''  
在model.py文件中我们定义了一个ResNet152网络
'''  
python model.py --log_dir='你存储tensorboard文件的路径'  
'''
会在指定的路径下生成一个以***系统当前时间命名的文件夹***



