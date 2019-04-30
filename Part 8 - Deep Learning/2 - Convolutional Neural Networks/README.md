
# Convolutional Neural Networks

## Plan of attack

- What are Convolutional Neural Networks ?
- Step 1 - Convolution Operation
- Step 1(b) - ReLU Layer
- Step 2 - Pooling
- Step 3 - Flattening
- Step 4 - Full connection
- Summary
- Bonus: Softmax & Cross-Entropy

---

## What are Convolutional Neural Networks ?

<img src="img/cnn_intro.png" width="400" height="200">
<br>
<img src="img/cnn_intro_2.png" width="400" height="200">

CNN input are images and can classify them to put a label ont it.
The computer understand input black and white or collored picture like a 2d pr 3d array of pixels.

<img src="img/cnn_intro_3.png" width="400" height="200">
<br>
<img src="img/cnn_intro_4.png" width="400" height="200">

#### To go further : 
[=> Gradient-based learning applied to document recognition by Yann LeCun (1998)](http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf)

---

## Step 1 - Convolution Operation

<img src="img/conv_function.png" width="400" height="200">

In the convolution operation, there are 3 elements : the input image, the feature detector and the feature map.
We calculate the feature map by multiplying parts of the input map and the feature detector (elements by elements and putting  the number of matches).

Each step/calculation (Square of 9 cases) it called a stride.

<img src="img/conv_0.png" width="400" height="200">
<br>
<img src="img/conv_1.png" width="400" height="200">
<br>
...

<br>
<img src="img/conv_2.png" width="400" height="200">
<br>
<img src="img/conv_3.png" width="400" height="200">
...

<br>
<img src="img/conv_4.png" width="400" height="200">

Doing this actions reduce the size of the image to make it easier and faster to process it.

Features are how the machine see elements in the picture. As you can see the feature detector has a pattern and the biggest number found in the feature map is when we found the max number of matches.

So at the end we will create many feature maps by using different features detectors (or filters).

The convolution operation help us to extract feature from a picture to using a feature detector to create different feature maps.

<img src="img/conv_5.png" width="400" height="200">

__Example of features__:

<img src="img/conv_example_1.png" width="400" height="200">
<br>
<img src="img/conv_example_2.png" width="400" height="200">
<br>
<img src="img/conv_example_3.png" width="400" height="200">

#### To go further : 
[=> Introduction to Convolutional Nerual Networks by Jianxin Wu (2017) ](https://pdfs.semanticscholar.org/450c/a19932fcef1ca6d0442cbf52fec38fb9d1e5.pdf)

---

## Step 1(b) - ReLU Layer

This an additionnal step of the convolution operation. After the previous step we will apply an activation function for example the rectifier.

<img src="img/reLU_1.png" width="400" height="200">

The rectifier help us to increase the non-linearity in the neural network because images themselves are highly non-linear (different element like borders, colors, pixels ...). 

__Example:__

<img src="img/reLU_2.png" width="400" height="200">

If we extract features of this image of buildings.

<img src="img/reLU_3_1.png" width="400" height="200">

We can see sometime some linearity between colors, like some parts of the image when the folowing colors are white -> grey -> dark or dark -> grey -> white

<img src="img/reLU_3.png" width="400" height="200">

The rectifier help us to break this kind of linearity.

<img src="img/reLU_4.png" width="400" height="200">

#### To go further : 
[=> Understanding Convolutional Neural Networks with A Mathematical Model by C.C Jay Kuo (2016)](https://arxiv.org/abs/1609.04112)


---
## Step 2 - Pooling

<img src="img/pooling_0.png" width="400" height="200">

Pooling help us to detecte the same feature however the element we want to extract is placed in the picture. For example the face of the cheetah above.

There are different types of pooling (min, max ...).

Let's take the example above and especially our feature map.

When we apply max pooling , we are doing like convolution applying it to strides and take the max value in order to get a pooled feature map.

<img src="img/pooling_1.png" width="400" height="200">

...
<br>

<img src="img/pooling_2.png" width="400" height="200">

We are still keeping our features (big numbers) but we are removing not important informations and also avoiding kind of distorsion.
We are also reducing the size and the number of parameters coming from our original image

#### Summary:

<img src="img/pooling_3.png" width="400" height="200">

#### Example:
[=> Interactive app - Detailed steps for an image](http://scs.ryerson.ca/~aharley/vis/conv/flat.html)

<img src="img/pooling_4.png" width="600" height="400">


#### To go further : 
[=> Evaluation of Pooling Operations in Convolutional Architectures for Object Recognition by Dominik Scherer (2010)](http://ais.uni-bonn.de/papers/icann2010_maxpool.pdf)


---
## Step 3 - Flattening

From the pooled feature map, we take the elements row by row and put in a column.

<img src="img/flattening_0.png" width="400" height="200">

The goal is to use these columns as vectors of input for a ANN (Artificial neural network)

<img src="img/flattening_1.png" width="400" height="200">

#### Summary:

<img src="img/flattening_2.png" width="400" height="200">

---

## Step 4 - Full connection

<img src="img/fc_0.png" width="400" height="200">

After flattening, we get the input values for our artificial neural network. But in this case neurons of each layer are __fully connected__ to each other (no weights with 0 as a value). In the output layer of our ANN, we have a neuron for each category of the classification.

<img src="img/fc_1.png" width="400" height="200">

Like a normal ANN, the information from the input values are going to be weighted propagated through synapse and neuron and give a probability for an output value (80% dog). 
Then the error is calculated using a loss function and the return is used to __adjust the weights but also (that's the difference) the features__.

<img src="img/fc_2.png" width="400" height="200">

What weights to assign for each ouput value ?

Depend of the value return at the ouput and the error, when the signal is back propagated the weights and neurons for specific features (element of the picture) to choose are the ones which are discrimanitive or descriptive for a category.

<img src="img/fc_3.png" width="400" height="200">

<img src="img/fc_4.png" width="400" height="200">

---

## Summary

<img src="img/summary_0.png" width="600" height="400">



---
## Bonus: Softmax & Cross-Entropy

---

## Practical Example

___

### Installation Tips

#### Installing Theano

Theano is an open source library for fast computations for GPU (graphics, more powerful) and CPU (your computer).

```bash
pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git
```

#### Installing Tensorflow

Tensorflow is another open source library developped by Google Brain used to develop neural network from scratch.

[Install Tensorflow from the website](https://www.tensorflow.org/versions/r0.12/get_started/os_setup.html)

```bash
pip install tensorflow

```

#### Installing Keras

Open source library to build neural network using few lines or codes.

```bash
pip install --upgrade keras
```
___

### Dataset 

Images of cats and dofs :)


---

### Data Processing

Nothing to do :)


---

### Building CNN model



```python

```


```python
# Convolutional Neural Network

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from IPython.display import display
from PIL import Image


# Initialising the CNN
classifier = Sequential()

# Step 1 - Convolution
classifier.add(Convolution2D(32, 3, 3, input_shape = (64, 64, 3), activation = 'relu'))

# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Adding a second convolutional layer
classifier.add(Convolution2D(32, 3, 3, activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full connection
classifier.add(Dense(output_dim = 128, activation = 'relu'))
classifier.add(Dense(output_dim = 1, activation = 'sigmoid'))

# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Part 2 - Fitting the CNN to the images

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

test_set = test_datagen.flow_from_directory('dataset/test_set',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')

classifier.fit_generator(training_set,
                         samples_per_epoch = 8000,
                         nb_epoch = 25,
                         validation_data = test_set,
                         nb_val_samples = 2000)
```

    Using TensorFlow backend.


    WARNING:tensorflow:From /usr/local/lib/python3.7/site-packages/tensorflow/python/framework/op_def_library.py:263: colocate_with (from tensorflow.python.framework.ops) is deprecated and will be removed in a future version.
    Instructions for updating:
    Colocations handled automatically by placer.


    /usr/local/lib/python3.7/site-packages/ipykernel_launcher.py:17: UserWarning: Update your `Conv2D` call to the Keras 2 API: `Conv2D(32, (3, 3), input_shape=(64, 64, 3..., activation="relu")`
    /usr/local/lib/python3.7/site-packages/ipykernel_launcher.py:23: UserWarning: Update your `Conv2D` call to the Keras 2 API: `Conv2D(32, (3, 3), activation="relu")`
    /usr/local/lib/python3.7/site-packages/ipykernel_launcher.py:30: UserWarning: Update your `Dense` call to the Keras 2 API: `Dense(activation="relu", units=128)`
    /usr/local/lib/python3.7/site-packages/ipykernel_launcher.py:31: UserWarning: Update your `Dense` call to the Keras 2 API: `Dense(activation="sigmoid", units=1)`


    Found 8000 images belonging to 2 classes.
    Found 2000 images belonging to 2 classes.
    WARNING:tensorflow:From /usr/local/lib/python3.7/site-packages/tensorflow/python/ops/math_ops.py:3066: to_int32 (from tensorflow.python.ops.math_ops) is deprecated and will be removed in a future version.
    Instructions for updating:
    Use tf.cast instead.


    /usr/local/lib/python3.7/site-packages/ipykernel_launcher.py:61: UserWarning: The semantics of the Keras 2 argument `steps_per_epoch` is not the same as the Keras 1 argument `samples_per_epoch`. `steps_per_epoch` is the number of batches to draw from the generator at each epoch. Basically steps_per_epoch = samples_per_epoch/batch_size. Similarly `nb_val_samples`->`validation_steps` and `val_samples`->`steps` arguments have changed. Update your method calls accordingly.
    /usr/local/lib/python3.7/site-packages/ipykernel_launcher.py:61: UserWarning: Update your `fit_generator` call to the Keras 2 API: `fit_generator(<keras_pre..., validation_data=<keras_pre..., steps_per_epoch=250, epochs=25, validation_steps=2000)`


    Epoch 1/25
    250/250 [==============================] - 434s 2s/step - loss: 0.6792 - acc: 0.5644 - val_loss: 0.6519 - val_acc: 0.6130
    Epoch 2/25
    250/250 [==============================] - 1541s 6s/step - loss: 0.6215 - acc: 0.6510 - val_loss: 0.5907 - val_acc: 0.6876
    Epoch 3/25
    250/250 [==============================] - 846s 3s/step - loss: 0.5864 - acc: 0.6896 - val_loss: 0.5732 - val_acc: 0.7170
    Epoch 4/25
    250/250 [==============================] - 1148s 5s/step - loss: 0.5565 - acc: 0.7090 - val_loss: 0.5354 - val_acc: 0.7360
    Epoch 5/25
    250/250 [==============================] - 471s 2s/step - loss: 0.5295 - acc: 0.7293 - val_loss: 0.5216 - val_acc: 0.7522
    Epoch 6/25
    250/250 [==============================] - 424s 2s/step - loss: 0.5065 - acc: 0.7456 - val_loss: 0.6438 - val_acc: 0.6684
    Epoch 7/25
    250/250 [==============================] - 448s 2s/step - loss: 0.4904 - acc: 0.7621 - val_loss: 0.4972 - val_acc: 0.7669
    Epoch 8/25
    250/250 [==============================] - 491s 2s/step - loss: 0.4760 - acc: 0.7707 - val_loss: 0.4894 - val_acc: 0.7654
    Epoch 9/25
    250/250 [==============================] - 412s 2s/step - loss: 0.4658 - acc: 0.7749 - val_loss: 0.4801 - val_acc: 0.7750
    Epoch 10/25
    250/250 [==============================] - 418s 2s/step - loss: 0.4426 - acc: 0.7899 - val_loss: 0.4884 - val_acc: 0.7721
    Epoch 11/25
    250/250 [==============================] - 455s 2s/step - loss: 0.4343 - acc: 0.7937 - val_loss: 0.5241 - val_acc: 0.7461
    Epoch 12/25
    250/250 [==============================] - 434s 2s/step - loss: 0.4315 - acc: 0.7996 - val_loss: 0.4554 - val_acc: 0.7880
    Epoch 13/25
    250/250 [==============================] - 419s 2s/step - loss: 0.4130 - acc: 0.8076 - val_loss: 0.5129 - val_acc: 0.7705
    Epoch 14/25
    250/250 [==============================] - 446s 2s/step - loss: 0.4143 - acc: 0.8065 - val_loss: 0.5067 - val_acc: 0.7465
    Epoch 15/25
    250/250 [==============================] - 402s 2s/step - loss: 0.4006 - acc: 0.8226 - val_loss: 0.4681 - val_acc: 0.7857
    Epoch 16/25
    250/250 [==============================] - 403s 2s/step - loss: 0.3844 - acc: 0.8245 - val_loss: 0.4799 - val_acc: 0.7987
    Epoch 17/25
    250/250 [==============================] - 414s 2s/step - loss: 0.3772 - acc: 0.8310 - val_loss: 0.4687 - val_acc: 0.7878
    Epoch 18/25
    250/250 [==============================] - 421s 2s/step - loss: 0.3692 - acc: 0.8313 - val_loss: 0.4723 - val_acc: 0.7948
    Epoch 19/25
    250/250 [==============================] - 414s 2s/step - loss: 0.3633 - acc: 0.8380 - val_loss: 0.5176 - val_acc: 0.7838
    Epoch 20/25
    250/250 [==============================] - 415s 2s/step - loss: 0.3527 - acc: 0.8425 - val_loss: 0.5178 - val_acc: 0.7783
    Epoch 21/25
    250/250 [==============================] - 423s 2s/step - loss: 0.3474 - acc: 0.8478 - val_loss: 0.4816 - val_acc: 0.7902
    Epoch 22/25
    250/250 [==============================] - 432s 2s/step - loss: 0.3447 - acc: 0.8479 - val_loss: 0.5011 - val_acc: 0.7968
    Epoch 23/25
    250/250 [==============================] - 1040s 4s/step - loss: 0.3335 - acc: 0.8501 - val_loss: 0.4701 - val_acc: 0.7935
    Epoch 24/25
    250/250 [==============================] - 411s 2s/step - loss: 0.3294 - acc: 0.8559 - val_loss: 0.4936 - val_acc: 0.7951
    Epoch 25/25
    250/250 [==============================] - 399s 2s/step - loss: 0.3161 - acc: 0.8600 - val_loss: 0.4904 - val_acc: 0.7863





    <keras.callbacks.History at 0x1261fcfd0>




```python

```
