import parameters as p
import utils
from Conv_NN import Conv_NN
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # disable tensorflow warnings
import tensorflow as tf
from tensorflow import keras
keras.backend.set_floatx(p.sdtypestr)
keras.utils.set_random_seed(p.seed)

class Standard_Conv_NN(Conv_NN):
    def __init__(self, model):
        self.conv_layers = utils.get_conv_layers(model)
        self.dense_layers = utils.get_dense_layers(model)
        # Weights
        self.W_dict = {}
        for i in self.conv_layers + self.dense_layers:
            self.W_dict[i] = tf.cast(model.layers[i].weights[0], dtype=p.sdtype)
        # Biases
        self.b_dict = {}
        for i in self.conv_layers + self.dense_layers:
            self.b_dict[i] = tf.cast(model.layers[i].weights[1], dtype=p.sdtype)
        # Other
        self.loss = keras.losses.CategoricalCrossentropy()
        self.iter = 0
        self.model = model
        self.mode = "standard"
        # Stats
        self.epochs = []
        self.times = []
        self.losses = []
        self.accuracies = []

    def grads(self, x, y): # calculating multiple gradients at once is faster
            # Compute gradients with respect to W, b and Us
            with tf.GradientTape() as t:
                t.watch(self.W_dict)
                t.watch(self.b_dict)
                out = self.loss(y, self.func(x, train=True))
                grads_W, grads_b = t.gradient(out, [self.W_dict, self.b_dict])
                # Casting
                grads_W = {layer: tf.cast(grad, dtype=p.sdtype) for (layer,grad) in grads_W.items()}
                grads_b = {layer: tf.cast(grad, dtype=p.sdtype) for (layer,grad) in grads_b.items()}
            return grads_W, grads_b
    
    def step(self, x_train, y_train, stepsize, *_): # ignores additional inputs (adaptive, tau)
        # Compute gradients
        grads_W, grads_b = self.grads(x_train, y_train)
        # Update biases
        self.b_dict = {layer: b - stepsize * grads_b[layer] for (layer,b) in self.b_dict.items()}
        # Update weights of dense layers
        self.W_dict = {layer: W - stepsize * grads_W[layer] for (layer,W) in self.W_dict.items()}
        # Increase iteration counter
        self.iter += 1
    
    def get_n_ops(self):
        # Compute total number of multiplications needed to evaluate the standard network with one sample
        n_ops = 0
        for i,l in enumerate(self.model.layers):
            if l.name[:6] == "conv2d":
                n = self.W_dict[i].shape
                hout = l.output.shape[1]
                gout = l.output.shape[2]
                n_ops += n[0]*n[1]*n[2] * hout*gout*n[3]
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
                n = self.W_dict[i].shape
                b = l.output.shape[-1]
                n_params += n[0]*n[1]*n[2]*n[3] + b
            elif l.name[:5] == "dense":
                nin = self.W_dict[i].shape[0]
                nout = self.W_dict[i].shape[1]
                b = l.output.shape[-1]
                n_params += nin*nout + b
        return n_params