from sklearn.datasets import load_boston
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import losses, optimizers
import numpy as np
def pltpicture(xlist,ylist):
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.scatter(xlist, ylist)
    plt.show()
def getrow(array,num):
    list=[]
    for i in array:
        list.append(i[num])
    return list
def doforone(data):   #归一化函数
    data_new=np.empty(shape=[506,0],dtype=float)
    for i in range(len(data[0])):
        list_new_temp=[]
        temp_list=getrow(data,i)
        max_num=max(temp_list)
        min_num=min(temp_list)
        for j in temp_list:
            list_new_temp.append((j-min_num)/(max_num-min_num))
        print()
        #np.column_stack(np.array(list_new_temp).reshape(-1,1), data_new)
        data_new=np.insert(data_new,0,list_new_temp,axis=1)
    data_new.tolist()
    print(len(data_new[0]))
    return data_new
def slice(data,target):
    print(data)
    train=data[0:426]

    test=data[426:]
    max_num = max(target)
    min_num = min(target)
    target=(target-min_num)/(max_num-min_num)
    print("&"*8)
    print(target)
    target_train=[[i] for i in target[0:426]]
    target_test=target[426:]
    return train,test,target_train,target_test
if __name__ == '__main__':
    epoch=50
    batch=32
    boston=load_boston(return_X_y=False)
    data=boston["data"]
    target=boston["target"]
    fearture_names=boston["feature_names"]
    DESCR=boston["DESCR"]
    # #画图的步骤
    # list_x=getrow(data,12)
    # pltpicture(list_x,target)
    # print(list_x)
    #归一化处理
    data=doforone(data)
    print("#"*8)
    data_new=[]
    for i in data:
        data_new.append(i.tolist())
    print(data_new)
    print(type(data_new))


    #划分训练集和测试集
    train, test, target_train, target_test=slice(data_new,target)
    print(train)
    print(len(train))
    print(target_train)

    #构建训练datesets
    train_dataset = tf.data.Dataset.from_tensor_slices((train,target_train))

    #构建测试dateset
    test_dataset = tf.data.Dataset.from_tensor_slices((train, target_train))
    #对训练集进行处理
    train_dataset.shuffle(buffer_size=100).batch(batch).repeat(count=10)
    print(train_dataset)



    criteon = losses.MSE
    optimizer=optimizers.Adam()
    model = tf.keras.Sequential(
        [

          tf.keras.layers.Dense(10,input_shape=(13,),activation="relu"),
            tf.keras.layers.Dense(10,activation="relu"),
            tf.keras.layers.Dense(10,activation="relu"),
            tf.keras.layers.Dense(1),

        ]
    )

    model.summary()
    model.compile(optimizer=optimizer,
                  loss=criteon,
                  metrics=['accuracy'])
    model.fit(train,target_train,epochs=400)
