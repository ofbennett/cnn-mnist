# Convolutional Neural Network - applied to MNIST

By Oscar Bennett, 2019

This is a [TensorFlow](https://www.tensorflow.org) implementation of a [convolutional neural network (CNN)](https://ujjwalkarn.me/2016/08/11/intuitive-explanation-convnets/) applied to the task of image recognition using the MNIST dataset. This is a toy example of a computer vision task which demonstrates the power of CNNs at extracting information from images. I've avoided using any high level APIs (like Keras) and implemented most of the functionality with low level operations from scratch in order to be explicit about how everything works. There is a nice explanation of how CNNs work [here](https://ujjwalkarn.me/2016/08/11/intuitive-explanation-convnets/).

I've also used TensorFlow's great visualisation tool [TensorBoard](https://www.tensorflow.org/tensorboard). This allows you to visualise the progress of training as well as visualise the whole computational graph you define in your code.

<img src=resources/tensorboard_example.png width=100%>

The [MNIST dataset](https://en.wikipedia.org/wiki/MNIST_database) is a large collection of images of hand written digits. Its a nice and simple dataset to apply data science and machine learning methods to in order to demonstrate their use and benchmark their performance. Here are some examples:

<img src=resources/MNIST_example1.png width=80%>
<img src=resources/MNIST_example2.png width=80%>

The task here is to predict the digit labels from the image data. This task is actually also a long running [Kaggle competition](https://www.kaggle.com/c/digit-recognizer/overview) run to help people experiment with simple computer vision techniques. The model in this repo once trained will reach approximately 99.5% accuracy which is close to what is thought to be the highest possible (~99.7%). Getting that last tiny performance boost to reach 99.7% usually requires an ensemble of multiple models.

The datasets used to train and test the model are provided by Kaggle and can be downloaded [here](https://www.kaggle.com/c/digit-recognizer/data). Download the `train.csv` and `test.csv` files and put them in the top directory of this repo.

To run on CPU, clone the repo and then execute the following commands:
```
> conda create -n CNN_MNIST python=3.7 pip
> source activate CNN_MNIST
> conda install tensorflow
> pip install -r requirements.txt
> python run.py
```

To run on a GPU (recommended if you can - model will train a lot faster) you will need an NVIDIA GPU with [CUDA installed and setup](https://docs.nvidia.com/cuda/#installation-guides). Execute the following commands:
```
> conda create -n CNN_MNIST_GPU python=3.7 pip
> source activate CNN_MNIST_GPU
> conda install tensorflow-gpu
> pip install -r requirements.txt
> python run.py
```

The above commands will setup the environment in [conda](https://docs.conda.io/en/latest/), load the data, define the computational graph which underlies the CNN, then train the model over multiple epochs until the validation performance stops improving. Finally it will load the best performing version of the model from training and use it to generate predictions for the Kaggle test set (saved to a `submission.csv` file).

The final trained TensorFlow model checkpoints are saved in a `model/` directory. The logs for TensorBoard are saved in a `tf_logs/` directory.

To view the training progress or the computational graph on TensorBoard type:
```
> tensorboard --logdir tf_logs
```
Then open the provided url in a browser.

The model hyperparameters are set at reasonably good values, but if you want to play with them they are defined at the top of the `run.py` file:

```
max_n_epochs = 50
patience = 5 # epochs to continue beyond last highest validation score
batch_size = 64
dropout_rate = 0.4
percent_data_for_training = 90
```

To improve the performance of the model I implemented a few basic model and training features such [batch normalization](https://arxiv.org/abs/1502.03167), [early stopping](https://en.wikipedia.org/wiki/Early_stopping), [dropout](http://jmlr.org/papers/volume15/srivastava14a/srivastava14a.pdf), and [data augmentation](https://medium.com/nanonets/how-to-use-deep-learning-when-you-have-limited-data-part-2-data-augmentation-c26971dc8ced). If you're curious about these techniques just follow the links to discover more. I've implemented each from scratch to demonstrate how they work (all in the `tools.py` file except for early stopping).

Batch Normalization ([paper](https://arxiv.org/abs/1502.03167)):
```python
def batch_normalize(Z):
    A = tf.Variable(1,name='A',dtype=tf.float32) # Trainable scale parameter
    B = tf.Variable(0,name='B',dtype=tf.float32) # Trainable offset parameter
    mu = tf.reduce_mean(Z)
    std = tf.math.reduce_std(Z)
    Z_norm = (Z - mu)/std
    Z_bn = (Z_norm * A) + B
    print('Applying batch normalization')
    return Z_bn
```

Dropout ([paper](http://jmlr.org/papers/volume15/srivastava14a/srivastava14a.pdf)):
```python
def dropout(Z,rate):
    scale = 1/(1-rate) # Used to ensure expectation of sum of layer output remains the same
    shape = tf.shape(Z)
    temp = tf.random.uniform(shape, dtype=Z.dtype)
    mask = tf.cast((temp >= rate),Z.dtype) # Random mask of 1s and 0s
    Z_DO = tf.cond(rate>0, lambda: Z * mask * scale, lambda: Z) # For efficiency, if rate=0 simply return Z
    print('Applying dropout')
    return Z_DO
```

Data Augmentation:
```python
def data_augmentation(X_batch):
    batch_size = X_batch.shape[0]
    rot_angle_deg = random.uniform(-15,15) # random rotations (degrees)
    shift = random.randint(-1,1) # random shifts (pixels)
    image_stack = np.copy(X_batch)
    image_stack_rot = ndimage.rotate(image_stack,rot_angle_deg,axes=(1,2),reshape=False,mode='reflect',order=3)
    image_stack_rot_shift = ndimage.shift(image_stack_rot,(0,shift,shift,0),mode='reflect',order=0)
    X_batch_aug = image_stack_rot_shift
    return X_batch_aug
```

The fact that a CNN like this can reach 99.5% accuracy is impressive and is testament to the power of CNNs in general. When trying to improve a model it can be instructive to look at the training or validation examples that the model is getting wrong. Here are some examples of the digits that the trained model misclassifies:

<img src=resources/misclassified_digits_example.png width=70%>

A sample of misclassified digits like this can be saved simply by setting `SHOW_INCORRECT = True` at the top of the `run.py` file.

As you can see the digits that the model struggles with are actually pretty weird and even a human would struggle to know what digit some of these images were meant to represent! Seems that even a powerful machine learning model struggles to know what you mean if your hand writing is terrible :) What this essentially shows us it that the model is approaching human level accuracy and that significant further gains in performance will be difficult to achieve.

I hope this repo helps to demonstrate the power of a CNN and how you can use low level TensorFlow to implement one. You can even implement extra features like batch normalization and dropout yourself. Have fun with it and please feel free to let me know about any suggestions or issues!
