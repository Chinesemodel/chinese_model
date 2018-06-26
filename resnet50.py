from __future__ import print_function
import tensorflow as tf
from PIL import Image
from keras.models import model_from_json, load_model
from keras.models import Sequential
from keras.layers import Dense, Activation
import numpy as np
from keras.layers import Conv2D, BatchNormalization, MaxPooling2D, Input,ZeroPadding2D
from keras.layers import AveragePooling2D, Flatten, GlobalMaxPooling2D
from keras import layers
import os
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
import keras.backend as K
from keras.layers import merge



data_dir = '../data1'
train_data_dir = os.path.join(data_dir, 'train')
test_data_dir = os.path.join(data_dir, 'test')


def train(model):
    """用以生成一个batch的图像数据，支持实时数据提升。训练时该函数会无限生成数据，直到达到规定的epoch次数为止。"""
    train_datagen = ImageDataGenerator(
       #缩放因子，用于对图片进行缩放
        rescale=1. / 255,
      #整数，数据提升时，图片随机转动的角度
        rotation_range=0.3,
        #浮点数，图片宽度的某个比例，数据提升时图片水平偏移的幅度
        width_shift_range=0.1,
        #浮点数，图片高度的某个比例，数据提升时图片竖直偏移的幅度
        height_shift_range=0.1
      )
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    #以文件夹路径为参数,生成经过数据提升/归一化后的数据,在一个无限循环中无限产生batch数据
    train_generator = train_datagen.flow_from_directory(
        #目标文件夹路径,对于每一个类,该文件夹都要包含一个子文件夹.子文件夹中任何JPG、PNG、BNP、PPM的图片都会被生成器使用
        train_data_dir,
        #整数tuple,默认为(256, 256). 图像将被resize成该尺寸
        target_size=(64, 64),
        #batch数据的大小,默认32
        batch_size=16,
        #颜色模式,为"grayscale","rgb"之一,默认为"rgb".代表这些图片是否会被转换为单通道或三通道的图片.
        color_mode="rgb",
       
        class_mode='categorical')
    validation_generator = test_datagen.flow_from_directory(
        test_data_dir,
        target_size=(64, 64),
        batch_size=16,
        color_mode="rgb",
        
                
        #categorical, binary, sparse或None之一. 默认为categorical. 该参数决定了返回的标签数组的形式,
        #categorical会返回2D的one-hot编码标签,
       # binary返回1D的二值标签.
       # sparse返回1D的整数标签,
       # 如果为None则不返回任何标签, 生成器将仅仅生成batch数据, 
        #这种情况在使用model.predict_generator()和model.evaluate_generator()等函数时会用到.
        class_mode='categorical')
    

        

    model.compile(
        #损失函数
        loss='categorical_crossentropy',
        #优化器
        optimizer='rmsprop',
       
        #指标列表metrics：对分类问题，我们一般将该列表设置为metrics=['accuracy']。
        #指标可以是一个预定义指标的名字,也可以是一个用户定制的函数.
        #指标函数应该返回单个张量,或一个完成metric_name - > metric_value映射的字典.
        metrics=['accuracy'])
     
    """
    利用Python的生成器，逐个生成数据的batch并进行训练。
    生成器与模型将并行执行以提高效率。
    例如，该函数允许我们在CPU上进行实时的数据提升，同时在GPU上进行模型训练
    """
        
    model.fit_generator(
        
       # 一个形如（inputs，targets）的tuple
        #一个形如（inputs, targets,sample_weight）的tuple。
       # 所有的返回值都应该包含相同数目的样本。生成器将无限在数据集上循环。
       # 每个epoch以经过模型的样本数达到samples_per_epoch时，记一个epoch结束
        
        train_generator,
        #整数，当生成器返回steps_per_epoch次数据时计一个epoch结束，执行下一个epoch
        steps_per_epoch=84,
        #整数，数据迭代的轮数                
        epochs=10)
    json_string = model.to_json()  
    open('architecture.json','w').write(json_string)    
    model.save_weights('weights.h5')

# 输入输出大小相同的模块, 接受一个张量为输入，输出一个张量
def Conv2d_BN(x, nb_filter, kernel_size, strides=(1, 1), padding='same', name=None):
    if name is not None:
        bn_name = name + '_bn'
        conv_name = name + '_conv'
    else:
        bn_name = None
        conv_name = None

    x = Conv2D(nb_filter, kernel_size, padding=padding, strides=strides, activation='relu', name=conv_name)(x)
    x = BatchNormalization(axis=3, name=bn_name)(x)
    return x


def identity_Block(inpt, nb_filter, kernel_size, strides=(1, 1), with_conv_shortcut=False):
    x = Conv2d_BN(inpt, nb_filter=nb_filter, kernel_size=kernel_size, strides=strides, padding='same')
    x = Conv2d_BN(x, nb_filter=nb_filter, kernel_size=kernel_size, padding='same')
    if with_conv_shortcut:
        shortcut = Conv2d_BN(inpt, nb_filter=nb_filter, strides=strides, kernel_size=kernel_size)
        x = layers.add([x, shortcut])
        return x
    else:
        x = layers.add([x, inpt])
        return x

def bottleneck_Block(inpt,nb_filters,strides=(1,1),with_conv_shortcut=False):
    k1,k2,k3=nb_filters
    x = Conv2d_BN(inpt, nb_filter=k1, kernel_size=1, strides=strides, padding='same')
    x = Conv2d_BN(x, nb_filter=k2, kernel_size=3, padding='same')
    x = Conv2d_BN(x, nb_filter=k3, kernel_size=1, padding='same')
    if with_conv_shortcut:
        shortcut = Conv2d_BN(inpt, nb_filter=k3, strides=strides, kernel_size=1)
        x = layers.add([x, shortcut])
        return x
    else:
        x = layers.add([x, inpt])
        return x


def my_resnet():
    inpt = Input(shape=(64, 64, 3))
    # inpt = Input(shape=(width, height, channel))
    # x = ZeroPadding2D((3, 3))(inpt)

    #conv1
    x = Conv2d_BN(inpt, nb_filter=64, kernel_size=(7, 7), strides=(2, 2), padding='same')
    # x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)

    #conv2_x
    x = bottleneck_Block(x, nb_filters=[64,64,256],strides=(1,1),with_conv_shortcut=True)
    x = bottleneck_Block(x, nb_filters=[64,64,256])
    x = bottleneck_Block(x, nb_filters=[64,64,256])

    #conv3_x
    x = bottleneck_Block(x, nb_filters=[128, 128, 512],strides=(2,2),with_conv_shortcut=True)
    x = bottleneck_Block(x, nb_filters=[128, 128, 512])
    x = bottleneck_Block(x, nb_filters=[128, 128, 512])
    x = bottleneck_Block(x, nb_filters=[128, 128, 512])

    #conv4_x
    x = bottleneck_Block(x, nb_filters=[256, 256, 1024],strides=(2,2),with_conv_shortcut=True)
    x = bottleneck_Block(x, nb_filters=[256, 256, 1024])
    x = bottleneck_Block(x, nb_filters=[256, 256, 1024])
    x = bottleneck_Block(x, nb_filters=[256, 256, 1024])
    x = bottleneck_Block(x, nb_filters=[256, 256, 1024])
    x = bottleneck_Block(x, nb_filters=[256, 256, 1024])

    #conv5_x
    x = bottleneck_Block(x, nb_filters=[512, 512, 2048], strides=(2, 2), with_conv_shortcut=True)
    x = bottleneck_Block(x, nb_filters=[512, 512, 2048])
    x = bottleneck_Block(x, nb_filters=[512, 512, 2048])

    x = AveragePooling2D(pool_size=(4, 4))(x)
    x = Flatten()(x)
    x = Dense(5, activation='softmax')(x)



    model = Model(inpt, x, name='My_Resnet')
    return model

model = my_resnet()
    # model = load_model("./model.h5")
train(model)
