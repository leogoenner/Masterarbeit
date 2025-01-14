import utils
import parameters as p
import tensorflow as tf
from functools import partial
from funcy import compose
from tensorflow import keras

class Conv_NN:
    def func(self, x, train=False):
        layer_funcs = []
        for i,l in enumerate(self.model.layers):
            if l.name[:6] == "conv2d":
                # Distinction between standard and tucker networks:
                if self.mode == "standard":
                    layer_funcs.insert(0, partial(tf.nn.conv2d, filters=self.W_dict[i], strides=l.strides, padding=l.padding.upper()))
                elif self.mode == "tucker":
                    layer_funcs.insert(0, partial(utils.tucker_conv, C=self.C_dict[i], Us=self.Us_dict[i], strides=l.strides, padding=l.padding.upper()))
                layer_funcs.insert(0, partial(tf.math.add, y=self.b_dict[i]))
                layer_funcs = utils.add_activation(layer_funcs, l, i)
            elif l.name[:13] == "max_pooling2d":
                layer_funcs.insert(0, partial(tf.nn.max_pool2d, ksize=l.pool_size, strides=l.strides, padding=l.padding.upper()))
            elif l.name[:7] == "flatten":
                layer_funcs.insert(0, partial(tf.reshape, shape=[-1, *l.output.shape[1::]]))
            elif l.name[:7] == "dropout":
                if train == False: # no dropout
                    pass
                else: # use dropout
                    seed = keras.random.randint([2],0,100)
                    layer_funcs.insert(0, partial(tf.nn.experimental.stateless_dropout, rate=l.rate, seed=seed)) # TODO: use keras.random.SeedGenerator? (see https://keras.io/api/random/random_ops/)
            elif l.name[:5] == "dense":
                layer_funcs.insert(0, partial(tf.linalg.matmul, b=tf.cast(self.W_dict[i], dtype=p.sdtype)))
                layer_funcs.insert(0, partial(tf.math.add, y=tf.cast(self.b_dict[i], dtype=p.sdtype)))
                layer_funcs = utils.add_activation(layer_funcs, l, i)
            else:
                raise NameError(f'Unknown layer type "{l.name}" of layer {i}.')
        return compose(*layer_funcs)(x)

    def evaluate(self, eval_data, printing=True):
        total_samples = 0
        total_correct = 0
        total_loss = 0
        for batch in eval_data:
            y_pred = self.func(batch[0])
            max_pred = tf.math.argmax(y_pred, axis=1)
            max_test = tf.math.argmax(batch[1], axis=1)
            correct = tf.cast(tf.math.equal(max_pred, max_test), dtype=p.sdtype)
            n_samples = tf.size(correct, out_type=p.sdtype)
            total_samples += n_samples
            total_correct += tf.math.reduce_sum(correct)
            total_loss += self.loss(batch[1], y_pred) * n_samples
        loss = total_loss / total_samples
        accuracy = total_correct / total_samples
        if printing == True:
            print(f"Loss {loss.numpy():.2f}, Accuracy {accuracy.numpy():.1%}")
        return loss.numpy(), accuracy.numpy()
    
    def save_stats(self, epoch, time, loss, acc, rks=None):
        self.epochs.append(epoch)
        self.times.append(time)
        self.losses.append(loss)
        self.accuracies.append(acc)
        if self.mode == "tucker":
            self.ranks.append(rks)