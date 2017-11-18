import pickle
import tensorflow as tf
from sklearn.model_selection import train_test_split
from alexnet import AlexNet
import time
from sklearn.utils import shuffle

nb_classes = 43
orig_size = 32
new_size = 227

# TODO: Load traffic signs data.
with open("train.p", mode='rb') as f:
    data = pickle.load(f)

# TODO: Split data into training and validation sets.

X_train, X_val, y_train, y_val = train_test_split(data['features'], data['labels'], test_size=0.93, random_state=0)

print(X_train.shape)

# TODO: Define placeholders and resize operation.
x = tf.placeholder(tf.float32, (None, orig_size, orig_size, 3))
y = tf.placeholder(tf.int32, (None))
resized = tf.image.resize_images(x, [new_size, new_size])

# TODO: pass placeholder as first argument to `AlexNet`.
fc7 = AlexNet(resized, feature_extract=True)
# NOTE: `tf.stop_gradient` prevents the gradient from flowing backwards
# past this point, keeping the weights before and up to `fc7` frozen.
# This also makes training faster, less work to do!
fc7 = tf.stop_gradient(fc7)



# TODO: Add the final layer for traffic sign classification.

shape = (fc7.get_shape().as_list()[-1], nb_classes)  # use this shape for the weight matrix
fc8W = tf.Variable(tf.truncated_normal(shape, stddev=1e-2))
fc8b = tf.Variable(tf.zeros([nb_classes]))

logits = tf.matmul(fc7, fc8W) + fc8b
#probs = tf.nn.softmax(logits)


# TODO: Define loss, training, accuracy operations.
# HINT: Look back at your traffic signs project solution, you may
# be able to reuse some the code.

# Optimizer:

one_hot_y = tf.one_hot(y, nb_classes)

cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=one_hot_y, logits=logits)
loss_operation = tf.reduce_mean(cross_entropy)
optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
training_operation = optimizer.minimize(loss_operation)

#Evaluate
saver = tf.train.Saver()
correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# TODO: Train and evaluate the feature extraction model.

batchSize = 128

with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())

    print("Training...")
    print()

    old_accuracy = 0.0
    accuracy_list = []
    trainAccuracy_list = []

    num_examples = len(X_train)

    startTime = time.time()
    for i in range(20):
        #train
        X_train, y_train = shuffle(X_train, y_train)
        for offset in range(0, num_examples, batchSize):
            end = offset + batchSize
            batch_x, batch_y = X_train[offset:end], y_train[offset:end]
            sess.run(training_operation,
                     feed_dict={x: batch_x, y: batch_y})

        #evaluate

        X_data = X_val
        y_data = y_val

        num_examples = len(X_data)
        total_accuracy = 0
        #sess = tf.get_default_session()
        for offset in range(0, num_examples, batchSize):
            batch_x, batch_y = X_data[offset:offset + batchSize], y_data[
                                                                                  offset:offset + batchSize]
            # print(batch_x.shape)
            # print(batch_y.shape)
            accuracy = sess.run(accuracy_operation,
                                feed_dict={x: batch_x, y: batch_y})
            total_accuracy += (accuracy * len(batch_x))
        validation_accuracy =  total_accuracy / num_examples


        print("EPOCH {} ...".format(i + 1))
        print("Validation Accuracy = {:.3f}".format(validation_accuracy))
        print()

        deltaAccuracy = validation_accuracy - old_accuracy
        numEpochs = i + 1

        old_accuracy = validation_accuracy

    stopTime = time.time()

