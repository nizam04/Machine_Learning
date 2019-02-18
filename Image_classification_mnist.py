'''
Tensorflow tutorial for image classification in MNIST dataset
'''
import tensorflow as tf
mnist = tf.keras.datasets.mnist

#Only needed to plot an image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

#Loading mnist data
(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

#Creating model
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape =(28,28)),
    tf.keras.layers.Dense(512, activation=tf.nn.relu),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])

#Needed to plot an image
'''img=mpimg.imread('/Users/sadieee04/Desktop/imagename.png')
imgplot = plt.imshow(img)
plt.show()'''

#Optimizer selection
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

#Training the model
model.fit(x_train, y_train, epochs=5)

#Testing/evaluating the model
model.evaluate(x_test, y_test)