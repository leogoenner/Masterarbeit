import parameters as p
from Standard_Conv_NN import Standard_Conv_NN
from Tucker_Conv_NN import Tucker_Conv_NN
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # disable tensorflow warnings
import tensorflow as tf
import numpy as np
import warnings
import time
import shutil
import pickle
from tensorflow import keras
from keras import layers
from einops import rearrange
from datetime import datetime

# Loading & saving of models and results
def save_model_and_results(instance, df_stats):
    # create folder
    date = datetime.today().strftime('%Y-%m-%d %H%M')
    save_folder = "models and results/" + date
    cwd = os.getcwd()
    os.makedirs(os.path.join(cwd, save_folder))
    # save cnn
    with open(save_folder + '/cnn.pickle', 'wb') as f:
        pickle.dump(instance, f)
    # save stats
    df_stats.to_csv(save_folder + '/stats.csv')
    # save parameter file
    shutil.copy('parameters.py', save_folder)

def load_model(load_name):
    load_folder = os.path.join("models and results/" + load_name)
    with open(load_folder + '/cnn.pickle', 'rb') as f:
        return pickle.load(f)


# Data handling
def load_data(name):
    num_classes = 10
    if name == "MNIST":
        # Load the data and split it between train and test sets
        (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
        # Make sure images have shape (28, 28, 1)
        x_train = np.expand_dims(x_train, -1)
        x_test = np.expand_dims(x_test, -1)
    elif name == "CIFAR10":
        # Load the data and split it between train and test sets
        (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    else:
        raise NameError(f'{name} is not a valid dataset name. Choose between "MNIST" and "CIFAR10".')
    # Scale images to the [0, 1] range
    x_train = x_train.astype(p.sdtypestr) / 255
    x_test = x_test.astype(p.sdtypestr) / 255
    # Convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes)
    return (x_train, y_train, x_test, y_test)

def get_meanstd(data, axis=(0)): # adapted from https://stackoverflow.com/a/50239245
    mean = np.mean(data, axis=axis, keepdims=True)
    std = np.sqrt(((data - mean)**2).mean(axis=axis, keepdims=True))
    return mean, std

def get_processed_data(name, batchsize, buffersize, val_split):
    # name: MNIST or CIFAR10
    # batchsize: number of samples per batch
    # buffersize: buffer for shuffling the data
    # val_split: share of data to be used as validation set
    # returns tf Datasets for training, validation and testing
    data = load_data(name)
    if name == "CIFAR10":
        mean, std = get_meanstd(data[0], axis=(0)) # calculate mean and standard deviation
        data = ((data[0]-mean)/std, data[1], (data[2]-mean)/std, data[3]) # standardize data
    x_train,y_train, x_test,y_test = data
    # Calculate number of samples for validation set
    train_samples = x_train.shape[0]
    val_samples = int(tf.math.floor(train_samples * val_split))
    # Create tf Datasets
    if val_samples > 0:
        train_dataset = tf.data.Dataset.from_tensor_slices((x_train[:-val_samples,:,:,:], y_train[:-val_samples,:]))
        val_dataset = tf.data.Dataset.from_tensor_slices((x_train[-val_samples:,:,:,:], y_train[-val_samples:,:]))
    else:
        train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    # Create batches
    train_data = train_dataset.shuffle(buffersize).batch(batchsize) # shuffle for each epoch
    if val_samples > 0:
        val_data = val_dataset.batch(batchsize)
    else:
        val_data = None
    test_data = test_dataset.batch(batchsize)
    return train_data, val_data, test_data


# Neural networks
def training(instance, epochs, training_data, test_data, mode, adaptive, tau, stepsize, decay_rate):
    # Check for previously executed epochs
    finished_epochs = max(instance.epochs + [0])
    # Actual training
    for e in range(finished_epochs, finished_epochs + epochs): # tf Datasets get reshuffled for each epoch by default
        print(f"Starting epoch {e+1}.")
        start_time = time.time()
        for el in training_data:
            instance.step(el[0], el[1], stepsize * (1-decay_rate)**e, adaptive, tau)
        print(f"Finished epoch {e+1}.")
        runtime = time.time() - start_time
        # Evaluate model
        loss, acc = instance.evaluate(test_data)
        if mode == "tucker":
            rks = {l: instance.C_dict[l].shape for l in instance.conv_layers} # save ranks
            if adaptive == True:
                for (l,r) in rks.items():
                    print(f"Kernel in layer {l} has ranks {r}.")
            instance.save_stats(e+1, runtime, loss, acc, rks)
        else:
            instance.save_stats(e+1, runtime, loss, acc)

def tucker_conv(input, C, Us, strides=1, padding="VALID"): # Steps as in https://upcommons.upc.edu/bitstream/handle/2117/106019/tfm_8.pdf, p.26
    # 1)
    Z_1 = tf.einsum('sr,bhgs->bhgr',Us[0],input) # equivalent to tf.matmul(input, Us[0])
    # 2)
    Z_2 = tf.nn.conv2d(Z_1,C,strides=strides,padding=padding)
    # 3)
    Y = tf.einsum('tr,bhgr->bhgt',Us[1],Z_2) # equivalent to tf.matmul(Z_2, Us[1], transpose_b=True)
    return Y

def add_activation(layer_funcs, l, i):
    # adds the activation function to the list layer_funcs
    if (l.activation == None) or (l.activation.__name__ == "linear"):
        pass
    elif l.activation.__name__ == "relu":
        layer_funcs.insert(0, tf.nn.relu)
    elif l.activation.__name__ == "softmax":
        layer_funcs.insert(0, tf.nn.softmax)
    else:
        raise NameError(f'Unknown activation function "{l.activation.__name__}" in layer {i}.')
    return layer_funcs

def create_keras_model(name):
    # name: "SimpleMNIST" or "CustomCIFAR10"
    # returns tensorflow NN
    num_classes = 10
    if name == "SimpleMNIST":
        input_shape = (28, 28, 1)
        # Taken from https://keras.io/examples/vision/mnist_convnet/
        model = keras.Sequential([
                keras.Input(shape=input_shape),
                layers.Conv2D(32, kernel_size=(3,3), activation="relu"),
                layers.MaxPooling2D(pool_size=(2,2)),
                layers.Conv2D(64, kernel_size=(3,3), activation="relu"),
                layers.MaxPooling2D(pool_size=(2,2)),
                layers.Flatten(),
                layers.Dropout(0.5),
                layers.Dense(num_classes, activation="softmax")])
        # model.summary()
    elif name == "CustomCIFAR10":
        input_shape = (32, 32, 3)
        # Adapted from https://www.kaggle.com/code/roblexnana/cifar10-with-cnn-for-beginer
        model = keras.Sequential([
                keras.Input(shape=input_shape),
                layers.Conv2D(64, kernel_size=(3,3), activation="relu", padding="same"),
                layers.Conv2D(128, kernel_size=(3,3), activation="relu", padding="same"),
                layers.MaxPooling2D(pool_size=(2,2)),
                layers.Dropout(0.25),
                layers.Conv2D(256, kernel_size=(3,3), activation="relu"),
                layers.Conv2D(256, kernel_size=(3,3), activation="relu"),
                layers.MaxPooling2D(pool_size=(2,2)),
                layers.Dropout(0.5),
                layers.Flatten(),
                layers.Dense(num_classes, activation="softmax")])
        # model.summary()
    else:
        raise NameError(f'{name} is not a valid model name. Choose between "SimpleMNIST" and "CustomCIFAR10".')
    return model

def create_model(model, mode):
    # model: tensorflow NN
    # mode: "tucker" or "standard"
    # returns instance of Standard_Conv_NN or Tucker_Conv_NN
    if mode == "standard":
        return Standard_Conv_NN(model)
    elif mode == "tucker":
        return Tucker_Conv_NN(model)
    else:
        raise NameError(f'{mode} is not a valid mode. Choose between "standard" and "tucker".')

def get_conv_layers(model):
    # model: tensorflow NN
    # returns list of indices of all conv2d layers
    conv_inds = []
    for i,l in enumerate(model.layers):
        if l.name[:6] == "conv2d":
            conv_inds.append(i)
    return conv_inds

def get_dense_layers(model):
    # model: tensorflow NN
    # returns list of indices of all dense layers
    dense_inds = []
    for i,l in enumerate(model.layers):
        if l.name[:5] == "dense":
            dense_inds.append(i)
    return dense_inds


# Tensor operations
def prod(T,M,i):
    # T: tensor
    # M: matrix
    # i: mode along which the product is computed
    # returns tensor
    if i == 0:
        return tf.einsum('ijkl,mi->mjkl',T,M)
    elif i == 1:
        return tf.einsum('ijkl,mj->imkl',T,M)
    elif i == 2:
        return tf.einsum('ijkl,mk->ijml',T,M)
    elif i == 3:
        return tf.einsum('ijkl,ml->ijkm',T,M)

def tucker_approx(T, tau=None, ranks=None):
    # T: tensorflow tensor
    # tau: relative tolerance
    # ranks: predefined ranks 
    # returns core (tensorflow tensor) and factors (list of tensorflow tensors) whose product approximates the original tensor
    T = tf.convert_to_tensor(T, dtype=p.sdtype)
    Ps = []
    if ranks != None:
        for i,j in enumerate(range(2,4)):
            S, P, Q = tf.linalg.svd(mat(j,T))
            Ps.append(P[:,0:ranks[i]])
            Q = Q[:,0:ranks[i]]
            mat_T = tf.linalg.matmul(tf.linalg.diag(S[0:ranks[i]]), Q, transpose_b=True)
            shp = list(T.shape)
            if Q.shape[1] < ranks[i]:
                warnings.warn(f"Warning: The chosen rank {ranks[i]} exceeds the product of all the other ranks of the kernel. This leads to unnecessary computational cost.")
            shp[j] = Q.shape[1]
            shp = tf.TensorShape(shp)
            T = ten(j, shp, mat_T)
    elif tau != None:
        norm_T = tf.norm(T)
        for j in range(2,4):
            S, P, Q = tf.linalg.svd(mat(j,T))
            k = T.shape[j]
            while tf.norm(S[k-1:]) <= tau/2 * norm_T: # because only 2 modes are reduced, we can replace d (=4) with 2
                if k == 1:
                    break # prevent infinite loop and do not let rank drop below one
                k -= 1
            Ps.append(P[:,0:k])
            Q = Q[:,0:k]
            mat_T = tf.linalg.matmul(tf.linalg.diag(S[0:k]), Q, transpose_b=True)
            shp = list(T.shape)
            shp[j] = k
            shp = tf.TensorShape(shp)
            T = ten(j, shp, mat_T)
    return T, Ps

def tensor_shape(d):
    # d: number of modes
    # returns string of the type 'a1 a2 a3 a4'
    A = d * ['a']
    B = [str(b) for b in list(range(d))]

    tensor_shape = [''.join([a,b]) for a,b in zip(A, B)]
    tensor_shape = ' '.join(tensor_shape)
    return tensor_shape

def matrix_shape(d, i):
    # d: number of modes
    # i: mode for unfolding
    # returns string of the type 'a2 (a1 a3 a4)'
    A = d * ['a']
    B = [str(b) for b in list(range(d))]
    B.pop(i)
    B.insert(0, str(i))
    A[1] = '(' + A[1]
    B[-1] = B[-1] + ')'
    matrix_shape = [''.join([a,b]) for a,b in zip(A, B)]
    matrix_shape = ' '.join(matrix_shape)
    return matrix_shape

def mat(i,T):
    # i: unfolding along i-th mode
    # T: tensor
    # returns matrix
    return rearrange(T, tensor_shape(T.ndim) + ' -> ' + matrix_shape(T.ndim,i))

def ten(i,shape,M):
    # i: inverse to unfolding along i-th mode
    # shape: tuple with tensor dimensions
    # M: matrix
    # returns tensor
    d = len(shape)
    input = dict(zip(tensor_shape(d).split(' '),list(shape)))
    return rearrange(M, matrix_shape(d,i) + ' -> ' + tensor_shape(d), **input)