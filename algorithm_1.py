import utils
import tensorflow as tf

def algorithm_1(inst, grads_Us, adaptive, tau, x_train, y_train, alpha, beta):
    # inst: Tucker_Conv_NN
    # grads_Us: gradients with respect to the factor matrices (dict containing lists of tensors) 
    # adaptive: whether to dynamically update the ranks
    # tau: maximum relative error for the new core tensor
    # x_train: features of the training data
    # y_train: labels of the training data
    # alpha: stepsize for basis matrices
    # beta: stepsize for core tensor

    Us = inst.Us_dict
    delta_Us = grads_Us
    Us_new = {layer: 2*[None] for layer in inst.conv_layers}
    products = {layer: 2*[None] for layer in inst.conv_layers}

    for l in inst.conv_layers:
        for i in range(2):
            if adaptive:
                Us_new[l][i], _ = tf.linalg.qr(tf.concat([Us[l][i], delta_Us[l][i]], axis=1))
            else:
                Us[l][i] = Us[l][i] - alpha * delta_Us[l][i]
                Us_new[l][i], _ = tf.linalg.qr(Us[l][i])
            products[l][i] = tf.transpose(Us_new[l][i]) @ Us[l][i]

    C_tildes = inst.C_dict

    for l in inst.conv_layers:
        for i,j in enumerate(range(2,4)): # Us_new[i] is the basis matrix for mode i+3 (mode i+2 with zero-indexing)
            C_tildes[l] = utils.prod(C_tildes[l],products[l][i],j)

    inst.C_dict = C_tildes # for correct gradient computation
    inst.Us_dict = Us_new # for correct gradient computation

    delta_Cs = inst.grads_C(x_train, y_train) # compute gradients w.r.t. C for all convolutional layers at once

    for l in inst.conv_layers:
        inst.C_dict[l] = C_tildes[l] - beta * delta_Cs[l]

        if adaptive:
            inst.C_dict[l], Us[l] = utils.tucker_approx(inst.C_dict[l], tau=tau)
            for i in range(2):
                Us[l][i] = inst.Us_dict[l][i] @ Us[l][i]
            inst.Us_dict[l] = Us[l]
        else:
            pass

    return None # in-place modification is used