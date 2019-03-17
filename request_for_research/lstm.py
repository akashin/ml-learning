import tensorflow as tf
import tensorflow.contrib.eager as tfe
import numpy as np

class LSTM(tf.keras.layers.Layer):
    """Long Short Term Memory network module."""
    
    def __init__(self):
        super(LSTM, self).__init__()
    
    def build(self, shapes):
        x_shape, h_shape, c_shape = shapes
        x_size = x_shape[-1]
        h_size = h_shape[-1]
        c_size = c_shape[-1]
        
        assert(h_size == c_size)
        
        # Forget gate.
        self.W_fg = self.add_variable("W_fg", shape=[c_size, x_size + h_size], 
                                     initializer="uniform")
        
        # Input gate and input.
        self.W_ig = self.add_variable("W_ig", shape=[c_size, x_size + h_size], 
                                     initializer="uniform")
        self.W_i = self.add_variable("W_i", shape=[c_size, x_size + h_size], 
                                     initializer="uniform")
        
        # Output gate.
        self.W_og = self.add_variable("W_og", shape=[c_size, x_size + h_size], 
                                     initializer="uniform")
        
        super(LSTM, self).build(shapes)
    
    def call(self, inputs):
        x, h, c = inputs
        
        xh = tf.concat([x, h], 1)
        
        fg = tf.sigmoid(tf.matmul(xh, self.W_fg, transpose_b=True))
        ig = tf.sigmoid(tf.matmul(xh, self.W_ig, transpose_b=True))
        i = tf.tanh(tf.matmul(xh, self.W_ig, transpose_b=True))
        nc = fg * c + ig * i
        
        og = tf.sigmoid(tf.matmul(xh, self.W_og, transpose_b=True))
        nh = og * tf.tanh(nc)
        
        return nh, nc
