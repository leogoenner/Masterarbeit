import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # disable tensorflow warnings
import tensorflow as tf

#_________________________ Change parameters here ___________________________
load = False
load_name = "2024-12-06 1050" # (only relevant if load == True)
save = True

dataset = "MNIST" # MNIST or CIFAR10
cnn = "SimpleMNIST" # SimpleMNIST or CustomCIFAR10
init = "normal" # normal or factorize
scale = 0.15 # scale for the initialization of C via a normal distribution (only relevant if mode == "tucker" and init == "normal")

# Define the initial ranks for each layer (only relevant if mode == "tucker")
if cnn == "SimpleMNIST":
    ranks = {0:(1,5), 2:(6,12)}
elif cnn == "CustomCIFAR10":
    ranks = {0:(3,12), 1:(12,24), 4:(24,32), 5:(32,32)}

mode = "tucker" # tucker or standard
adaptive = True # (only relevant if mode == "tucker")
tau = 1e-2 # relative tolerance for tucker approximation (only relevant if mode == "tucker" and adaptive == True)
epochs = 5
accuracy = "single" # single or double
stepsize = 1e-1
decay_rate = 1.5e-1 # after each epoch: new_stepsize = (1-decay_rate) * old_stepsize

batchsize = 128
shuffle_buffersize = 100
validation_split = 0.0

seed = 7
#____________________________________________________________________________


# Do not change the following lines
if accuracy == "single":
    sdtype = tf.float32
    sdtypestr = "float32"
elif accuracy == "double":
    sdtype = tf.float64
    sdtypestr = "float64"