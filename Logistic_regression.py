import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split


#data process
data_train = pd.read_csv ("./Data/train.csv")
data_y = data_train["label"]
data_x = data_train.drop(["label"],axis=1)
test_x = pd.read_csv ("./Data/test.csv")

#print (data_x)
#print (data_y)
data_y = np.array(data_y).reshape(len(data_y),-1)
enc = OneHotEncoder()
enc.fit(data_y)
targets = enc.transform(data_y).toarray()

train_x, valid_x, train_y, valid_y = train_test_split(data_x, targets, test_size= 0.2)


#Logistic regression model with tensorflow
epochs = 2
L_rate = 0.003
batch_size = 5000

x = tf.placeholder(tf.float32, [None,train_x.shape[1]])
y_ = tf.placeholder(tf.float32, [None,10])

W = tf.Variable(tf.truncated_normal([train_x.shape[1],10],stddev=0.1))
b = tf.Variable(tf.truncated_normal([10]))
y = tf.nn.softmax(tf.matmul(x,W) + b)

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_))
train_step  = tf.train.GradientDescentOptimizer(learning_rate= L_rate).minimize(cross_entropy)
predict_answer = tf.argmax(y,1)
init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)
epoch = 1
best_result = 0
while epoch <= epochs:
    print("epoch of training", epoch)
    for i in range(0, train_x.shape[0], batch_size):
        sess.run(train_step, feed_dict={x: train_x[i:i + batch_size], y_: train_y[i:i + batch_size]})
        loss = sess.run(cross_entropy, feed_dict={x: train_x[i:i + batch_size], y_: train_y[i:i + batch_size]})

        # predict accuracy
        if epoch % 2 == 0:
            same_prediction = tf.equal(tf.argmax(y_, 1), tf.argmax(y, 1))
            accuracy = tf.reduce_mean(tf.cast(same_prediction, tf.float32))

            accuracy_train = sess.run(accuracy, feed_dict={x: train_x, y_: train_y})
            accuracy_valid = sess.run(accuracy, feed_dict={x: valid_x, y_: valid_y})
            print("==================================================")
            print("Batch", i, "~", i + batch_size, "Of", epoch, "epochs")
            print("Accuracy on train set", accuracy_train)
            print("Accuracy on validate set", accuracy_valid, ", Loss = ", loss)
            print("Current best accuracy", best_result)

            if best_result < accuracy_valid:
                best_result = accuracy_valid
                predict_result = sess.run(predict_answer, feed_dict={x: test_x})

    epoch += 1
ImageId=range(1,28001)

print("The Final best accuracy: ", best_result)
evaluation = pd.DataFrame({'ImageId' : list(ImageId), 'Label':predict_result})
#valuation = pd.DataFrame(ImageId, columns=['ImageId'])
#evaluation = pd.DataFrame(predict_result, columns=['label'])
evaluation.to_csv("submission_2.csv",index = False)
