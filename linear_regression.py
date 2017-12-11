import tensorflow as tf
import numpy
import matplotlib.pyplot as plt
random_range = numpy.random

#Assigning values to some parameters
learning_rate = 0.01
step_size = 50
epochs = 1000

#Training Data
#train_X = numpy.asarray([1.2,7.2,1.3,7.9,10.657,12.458,2.365,8.54,9.145,10.187,7.1])
#train_Y = numpy.asarray([2.5,4.7,3.1,4.2,5.10,4.1,7.12,9.14,9.52,10.45,7.65])

#The data for train and test can be done using random function as well
train_X = numpy.asarray(random_range.randn(150))
train_Y = numpy.asarray(random_range.randn(150))

samples = train_X.shape[0]

# Input to the Graph
X = tf.placeholder("float")
Y = tf.placeholder("float")

# Setting the weights for the model
W = tf.Variable(random_range.randn(), name="weight")
b = tf.Variable(random_range.randn(), name="bias")

linear_model = tf.add(tf.multiply(X, W), b)

# Mean squared error
cost = tf.reduce_sum(tf.pow(linear_model-Y, 2))/(2*samples)

# Gradient descent
gradient_descent_optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# Initialize the variables to the default values
initialization = tf.global_variables_initializer()

# Start training
with tf.Session() as session:
    session.run(initialization)

    # Fit all training data
    for epoch in range(epochs):
        for (x, y) in zip(train_X, train_Y):
            session.run(gradient_descent_optimizer, feed_dict={X: x, Y: y})

        #Display logs per epoch step
        if (epoch+1) % step_size == 0:
            c = session.run(cost, feed_dict={X: train_X, Y:train_Y})
            print("Epoch:",'%04d' % (epoch+1), "cost=", "{:.9f}".format(c), "W=", session.run(W), "b=", session.run(b))
    print("-------------------------------------------------------")
    print("Optimization Finished!")
    print("-------------------------------------------------------")
    training_cost = session.run(cost, feed_dict={X: train_X, Y: train_Y})
    print("Training cost=", training_cost, "W=", session.run(W), "b=", session.run(b), '\n')

    #PLotting a graphs
    plt.plot(train_X, train_Y, 'ro', label='Data')
    plt.plot(train_X, session.run(W) * train_X + session.run(b), label='Fitted line')
    plt.legend()
    plt.show()