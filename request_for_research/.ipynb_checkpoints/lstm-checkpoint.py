import tensorflow as tf
import tensorflow.contrib.eager as tfe
import numpy as np

class GRU(tf.keras.layers.Layer):
    """Gated recurrent unit."""
    
    def __init__(self, hidden_size):
        self.hidden_size = hidden_size
        # self.hidden_state = tfe.Variable(np.zeros(self.hidden_size), dtype=tf.float32)
        super(GRU, self).__init__()
    
    def build(self, input_shape):
        print(input_shape)
        input_size = input_shape[-1]
        self.W_z = self.add_variable("W_z", shape=[self.hidden_size, self.hidden_size + input_size], 
                                     initializer="uniform")
        self.W_r = self.add_variable("W_r", shape=[self.hidden_size, self.hidden_size + input_size],
                                     initializer="uniform")
        self.W = self.add_variable("W", shape=[self.hidden_size, self.hidden_size + input_size],
                                   initializer="uniform")
    
    def call(self, input):
        print(input.dtype)
        print(self.hidden_state.dtype)
        z = tf.sigmoid(self.W_z * tf.stack([self.hidden_state, input]))
        r = tf.sigmoid(self.W_z * tf.stack([self.hidden_state, input]))
        h_hat = tf.tanh(self.W * tf.stack([r * self.hidden_state, input]))
        self.hidden_state = (1 - z) * self.hidden_state + z * h_hat
        return self.hidden_state
    
    #def compute_output_shape(self, input_shape):
        #return (input_shape[0], self.hidden_size)