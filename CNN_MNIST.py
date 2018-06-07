import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
import sklearn.metrics
import os
from scipy.ndimage.interpolation import rotate, shift, zoom
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns

#Data pre process

data_train = pd.read_csv ("Normal_trainx.csv")
data_train2 = pd.read_csv ("jitter_coldelete_trainx.csv")
data_train3 = pd.read_csv ("jitter_rowdelete_trainx.csv")
data_train4 = pd.read_csv ("jitter_nodelet__trainx.csv")
data_y = data_train["label"].values
data_train = data_train.drop(["label"],axis=1).values
data_train2 = data_train2.drop(["label"],axis=1).values
data_train3 = data_train3.drop(["label"],axis=1).values
data_train4 = data_train4.drop(["label"],axis=1).values
test_x = pd.read_csv ("./Data/test.csv")
test_x = test_x/255

targets = LabelBinarizer().fit_transform(data_y)
targets_np = np.copy(targets)
targets2 = np.concatenate((targets,targets_np),axis = 0)
targets3 = np.concatenate((targets2,targets_np),axis = 0)
targets4 = np.concatenate((targets3,targets_np),axis = 0)

data_x = np.concatenate((data_train,data_train2),axis = 0)
data_x = np.concatenate((data_x,data_train3),axis = 0)
data_x = np.concatenate((data_x,data_train4),axis = 0)

data_x = data_x/255
targets4 = targets4/255

#Seperate data
train_x, valid_x, train_y, valid_y = train_test_split(data_x, targets4, test_size= 0.002)


#Convolution neural network model with tensorflow
x = tf.placeholder(tf.float32, [None,train_x.shape[1]])
y_ = tf.placeholder(tf.float32, [None,10])
prvent_overfit = tf.placeholder(tf.float32)
Image_x = tf.reshape(x, [-1,28,28,1])

#First Convolution layer
W_conv1 = tf.Variable(tf.truncated_normal([5, 5, 1, 32],stddev = 0.1))
#b_conv1 = tf.Variable(tf.truncated_normal([32]))
b_conv1 = tf.constant(0.1,shape=[32])
First_conv = tf.nn.relu(tf.nn.conv2d(Image_x, W_conv1, strides=[1,1,1,1], padding='SAME')+b_conv1)
First_pool = tf.nn.max_pool(First_conv, ksize=[1,2,2,1], strides = [1,2,2,1], padding = 'SAME')

#Second Convolution layer
W_conv2 = tf.Variable(tf.truncated_normal([5,5,32,64],stddev = 0.1))
#b_conv2 = tf.Variable(tf.truncated_normal([64]))
b_conv2 = tf.constant(0.1,shape=[64])
Second_conv = tf.nn.relu(tf.nn.conv2d(First_pool, W_conv2, strides=[1,1,1,1], padding='SAME')+b_conv2)
Second_pool = tf.nn.max_pool(Second_conv, ksize=[1,2,2,1], strides = [1,2,2,1], padding = 'SAME')


#First fully connect layer
W_fc1 = tf.Variable(tf.truncated_normal([7*7*64,1024],stddev=0.1))
#b_fc1 = tf.Variable(tf.truncated_normal([1024]))
b_fc1 = tf.constant(0.1,shape=[1024])
Second_pool_flat= tf.reshape(Second_pool,[-1, 7*7*64])
y_fc1 = tf.nn.softmax(tf.matmul(Second_pool_flat,W_fc1) + b_fc1)
y_fc1_drop = tf.nn.dropout(y_fc1, prvent_overfit)

#Second fully connect layer
W_fc2 = tf.Variable(tf.truncated_normal([1024,10],stddev=0.1))
#b_fc2 = tf.Variable(tf.truncated_normal([10]))
b_fc2 = tf.constant(0.1,shape=[10])
predict_y = tf.nn.softmax(tf.matmul(y_fc1,W_fc2)+b_fc2)


#train algorithm
epochs = 40
L_rate = 0.001
batch_size = 100
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=predict_y))
train_step = tf.train.AdamOptimizer(L_rate).minimize(cross_entropy)
#train_step  = tf.train.GradientDescentOptimizer(learning_rate= L_rate).minimize(cross_entropy)
predict_answer = tf.argmax(predict_y,1)

#Set GPU
gpu_options = tf.GPUOptions(allow_growth=True)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))      

init = tf.global_variables_initializer()


#predict
same_prediction = tf.equal(tf.argmax(y_, 1), tf.argmax(predict_y, 1))
accuracy = tf.reduce_mean(tf.cast(same_prediction, tf.float32))

#restore model 
#saver = tf.train.Saver()
sess.run(init)
#saver.restore(sess, "/tmp/model.ckpt")
#print("Model restored.")
epoch = 1
best_result = 0
count = 0
loss_temp = 0
while epoch <= epochs:
    print("epoch of training", epoch)
    for i in range(0, train_x.shape[0], batch_size):
        train_datax = train_x[i:i + batch_size]
        train_datay = train_y[i:i + batch_size]
        
        sess.run(train_step, feed_dict={x: train_datax, y_: train_datay, prvent_overfit:0.5 })


        # when each 10 epoch show each batch accuracy
        if epoch % 10 == 0:

            loss = sess.run(cross_entropy, feed_dict={x: train_datax, y_: train_datay ,prvent_overfit:0.5})
            if loss > loss_temp:
                count +=1
                if count == 10:
                    L_rate = L_rate/1.25
                    print("L_rate/2 = ", L_rate)
                    if L_rate < 0.0000001:
                        L_rate = 0.0000001
                        print("L_rate Keep")
                    count =0
            loss_temp = loss
            #accuracy_train = sess.run(accuracy, feed_dict={x: train_x, y_: train_y})
            accuracy_valid = sess.run(accuracy, feed_dict={x: valid_x, y_: valid_y, prvent_overfit:0.5})
            print("==================================================")
            print("Batch", i, "~", i + batch_size, "Of", epoch, "epochs")
            #print("Accuracy on train set", accuracy_train)
            print("Accuracy on validate set", accuracy_valid, ", Loss = ", loss)
            print("Current best accuracy", best_result)

            if best_result < accuracy_valid:
                best_result = accuracy_valid

    #when each epoch 10 save the model
    if epoch % 10 == 0:
        save_path = saver.save(sess, "/tmp/model.ckpt")
        print("Model saved in path: %s" % save_path)
                
    epoch += 1

#show confusion matrix

y_predict_confuse_=[]
valida_y_confuse =[]
y_predict_confuse = sess.run(predict_answer, feed_dict={x: valid_x})
y_predict_confuse_.extend(y_predict_confuse)
valida_y_confuse.extend(np.argmax(valid_y,1))
cnf_matrix = sklearn.metrics.confusion_matrix( y_predict_confuse_, valida_y_confuse).astype(np.float32)
labels_array = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
fig, ax = plt.subplots(1,figsize=(10,10))
ax = sns.heatmap(cnf_matrix, ax=ax, cmap=plt.cm.Greens, annot=True)
ax.set_xticklabels(labels_array)
ax.set_yticklabels(labels_array)
plt.title('Confusion matrix of validation set')
plt.ylabel('True digit')
plt.xlabel('Predicted digit')
plt.show();

#predict answer for kaggle submission
predict_=[]
for j in range(0,test_x.shape[0], 500):
    predict_result = sess.run(predict_answer, feed_dict={x: test_x[j:j+500]})
    predict_.extend(predict_result)

ImageId=range(1,len(predict_)+1)
print("The Final best accuracy: ", best_result)
evaluation = pd.DataFrame({'ImageId' : list(ImageId), 'Label':predict_})
evaluation.to_csv("submission_2.csv",index = False)

