import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from scipy.ndimage.interpolation import rotate, shift, zoom
from scipy.misc import imresize
import matplotlib.pyplot as plt
import matplotlib.cm as cm

data_train = pd.read_csv ("./Data/train.csv")
data_y = data_train["label"]
data_x = data_train.drop(["label"],axis=1)
#est_x = pd.read_csv ("./Data/test.csv")
#test_x = test_x/255


#targets = LabelBinarizer().fit_transform(data_y)
#data_x = data_x/255
#targets = targets/255

#train_x, valid_x, train_y, valid_y = train_test_split(data_x, targets, test_size= 0.02)


def rand_jitter(temp):
    temp2 = temp
    temp2[np.random.randint(0,28,1), :] = 0
    temp2 = shift(temp2, shift=(np.random.randint(-3,3,2)))
    temp2 = rotate(temp2, angle = np.random.randint(-20,20,1), reshape=False)
    return temp2

def rand_jitter2(temp):
    temp2 = temp
    temp2[:, np.random.randint(0,28,1)] = 0
    temp2 = shift(temp2, shift=(np.random.randint(-3,3,2)))
    temp2 = rotate(temp2, angle = np.random.randint(-20,20,1), reshape=False)
    return temp2

def rand_jitter3(temp):
    temp2 = temp
    temp2 = shift(temp2, shift=(np.random.randint(-3,3,2)))
    temp2 = rotate(temp2, angle = np.random.randint(-20,20,1), reshape=False)
    return temp2

train_x_array = data_x.values
train_datax_array = np.copy(train_x_array)
train_datax_array = train_datax_array.reshape(train_datax_array.shape[0],28,28,1)
train_datax_array2 = np.copy(train_datax_array) 
train_datax_array3 = np.copy(train_datax_array) 
train_datay = np.copy(data_y)
train_datay2 = np.copy(data_y)
train_datay3 = np.copy(data_y)
train_datay4 = np.copy(data_y)

for j in range(0,train_datax_array.shape[0]):
    train_datax_array[j,:,:,0] = rand_jitter(train_datax_array[j,:,:,0])
    train_datax_array2[j,:,:,0] = rand_jitter2(train_datax_array2[j,:,:,0])
    train_datax_array3[j,:,:,0] = rand_jitter3(train_datax_array3[j,:,:,0])

train_x_array = train_x_array.reshape(-1,28,28,1)
plt.figure()
plt.subplot(2,2,1)
g=plt.imshow(train_x_array[0,:,:,0])

plt.subplot(2,2,2)
g=plt.imshow(train_datax_array[0,:,:,0])

plt.subplot(2,2,3)
g=plt.imshow(train_datax_array2[0,:,:,0])

plt.subplot(2,2,4)
g=plt.imshow(train_datax_array3[0,:,:,0])

plt.show()

train_x_array = train_x_array.reshape([-1,28*28*1])
train_datax_array = train_datax_array.reshape([-1,28*28*1])
train_datax_array2 = train_datax_array2.reshape([-1,28*28*1])
train_datax_array3 = train_datax_array3.reshape([-1,28*28*1])


train_y1=np.concatenate((train_datay,train_datay2),axis = 0)
train_y2=np.concatenate((train_y1,train_datay3),axis = 0)
data_y = np.concatenate((train_y2,train_datay4),axis = 0)

train_x3 = np.concatenate((train_x_array,train_datax_array),axis = 0)
train_x4 = np.concatenate((train_x3,train_datax_array2),axis = 0)
data_x = np.concatenate((train_x4,train_datax_array3),axis = 0)

#print(data_y.shape)
#print(data_x.shape)
data_save_x = pd.DataFrame(train_x_array)
data_save_x1 = pd.DataFrame(train_datax_array)
data_save_x2 = pd.DataFrame(train_datax_array2)
data_save_x3 = pd.DataFrame(train_datax_array3)
data_save_x["label"] = train_datay
data_save_x1["label"] = train_datay
data_save_x2["label"] = train_datay
data_save_x3["label"] = train_datay
data_save_x.to_csv("Normal_trainx.csv", index =False)
data_save_x1.to_csv("jitter_rowdelete_trainx.csv", index =False)
data_save_x2.to_csv("jitter_coldelete_trainx.csv", index =False)
data_save_x3.to_csv("jitter_nodelet__trainx.csv", index =False)
