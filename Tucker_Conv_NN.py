import parameters as p
import utils
from Conv_NN import Conv_NN
from algorithm_1 import algorithm_1
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # disable tensorflow warnings
import numpy as np
import tensorflow as tf
import scipy as sp
import random
from tensorflow import keras
keras.backend.set_floatx(p.sdtypestr)
keras.utils.set_random_seed(p.seed) # also sets the NumPy seed

class Tucker_Conv_NN(Conv_NN):
    def __init__(self, model):
        # Convolutional layer weights
        self.conv_layers = utils.get_conv_layers(model)
        self.C_dict = {}
        self.Us_dict = {} # will contain the basis matrices for the last two modes for each layer
        if p.init == "factorize":
            for i in self.conv_layers:
                C, Us = utils.tucker_approx(model.layers[i].weights[0], ranks=p.ranks[i])
                self.C_dict[i] = C
                self.Us_dict[i] = Us
        elif p.init == "normal":
            for i in self.conv_layers:
                shp = list(model.layers[i].weights[0].shape)
                shp_core = list(shp)
                shp_core[2:4] = p.ranks[i]
                C = np.random.normal(0.0, p.scale, shp_core)
                C = tf.convert_to_tensor(C, dtype=p.sdtypestr)
                Us = []
                for j,k in enumerate(range(2,4)):
                    if shp[k] > 1:
                        U = sp.stats.ortho_group.rvs(shp[k])[:, :p.ranks[i][j]]
                    else:
                        U = np.array(random.sample([-1,1], 1), ndmin=2)
                    Us.append(tf.convert_to_tensor(U, dtype=p.sdtypestr))
                self.C_dict[i] = C
                self.Us_dict[i] = Us
        # Dense layer weights
        self.dense_layers = utils.get_dense_layers(model)
        self.W_dict = {}
        for i in self.dense_layers:
            self.W_dict[i] = tf.cast(model.layers[i].weights[0], dtype=p.sdtype)
        # Biases
        self.b_dict = {}
        for i in self.conv_layers + self.dense_layers:
            self.b_dict[i] = tf.cast(model.layers[i].weights[1], dtype=p.sdtype)
        # Other
        self.loss = keras.losses.CategoricalCrossentropy()
        self.iter = 0
        self.model = model
        self.mode = "tucker"
        # Stats
        self.epochs = []
        self.times = []
        self.losses = []
        self.accuracies = []
        self.ranks = []
        
    def grads_C(self, x, y):
        # Compute gradients with respect to C
        with tf.GradientTape() as t:
            t.watch(self.C_dict)
            out = self.loss(y, self.func(x, train=True))
            grads = t.gradient(out, self.C_dict)
            grads = {layer: tf.cast(grad, dtype=p.sdtype) for (layer,grad) in grads.items()}
        return grads
    
    def grads(self, x, y): # calculating multiple gradients at once is faster
        # Compute gradients with respect to W, b and Us
        with tf.GradientTape() as t:
            t.watch(self.W_dict)
            t.watch(self.b_dict)
            t.watch(self.Us_dict)
            out = self.loss(y, self.func(x, train=True))
            grads_W, grads_b, grads_Us = t.gradient(out, [self.W_dict, self.b_dict, self.Us_dict])
            # Casting
            grads_W = {layer: tf.cast(grad, dtype=p.sdtype) for (layer,grad) in grads_W.items()}
            grads_b = {layer: tf.cast(grad, dtype=p.sdtype) for (layer,grad) in grads_b.items()}
            grads_Us = {layer: [tf.cast(U, dtype=p.sdtype) for U in grads] for (layer,grads) in grads_Us.items()}
        return grads_W, grads_b, grads_Us
    
    def step(self, x_train, y_train, stepsize, adaptive, tau):
        # Compute gradients
        grads_W, grads_b, grads_Us = self.grads(x_train, y_train)
        # Update biases
        self.b_dict = {layer: b - stepsize * grads_b[layer] for (layer,b) in self.b_dict.items()}
        # Update weights of dense layers
        self.W_dict = {layer: W - stepsize * grads_W[layer] for (layer,W) in self.W_dict.items()}
        # Update weights of convolutional layers via algorithm 1
        algorithm_1(self, grads_Us, adaptive, tau, x_train, y_train, stepsize, stepsize)
        # Increase iteration counter
        self.iter += 1

    def get_n_ops(self):
        # Compute total number of multiplications needed to evaluate the tucker network with one sample
        n_ops = 0
        for i,l in enumerate(self.model.layers):
            if l.name[:6] == "conv2d":
                r = self.C_dict[i].shape
                n = list(r[0:2]) + [U.shape[0] for U in self.Us_dict[i]]
                h = l.input.shape[1]
                g = l.input.shape[2]
                hout = l.output.shape[1]
                gout = l.output.shape[2]
                n_ops += n[2]*r[2]*h*g + r[0]*r[1]*r[2]*r[3]*hout*gout + r[3]*n[3]*hout*gout
            elif l.name[:5] == "dense":
                nin = self.W_dict[i].shape[0]
                nout = self.W_dict[i].shape[1]
                n_ops += nin*nout
        return n_ops
    
    def get_n_params(self):
        # Compute total number of parameters
        n_params = 0
        for i,l in enumerate(self.model.layers):
            if l.name[:6] == "conv2d":
                r = self.C_dict[i].shape
                n = list(r[0:2]) + [U.shape[0] for U in self.Us_dict[i]]
                b = l.output.shape[-1]
                n_params += r[0]*r[1]*r[2]*r[3] + n[2]*r[2] + n[3]*r[3] + b
            elif l.name[:5] == "dense":
                nin = self.W_dict[i].shape[0]
                nout = self.W_dict[i].shape[1]
                b = l.output.shape[-1]
                n_params += nin*nout + b
        return n_params