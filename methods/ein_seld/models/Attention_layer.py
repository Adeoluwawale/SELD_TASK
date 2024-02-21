

import tensorflow as tf
class DotProductAttention(tf.keras.layers.Layer):
    def __init__(self):
        super(DotProductAttention, self).__init__()

    def build(self, input_shape):
        self.W_q = self.add_weight(name='W_q', shape=(input_shape[-1], input_shape[-1]))
        self.W_k = self.add_weight(name='W_k', shape=(input_shape[-1], input_shape[-1]))
        self.b = self.add_weight(name='b', shape=(input_shape[-1],))

    def call(self, query, key):
        q = tf.matmul(query, self.W_q)
        k = tf.matmul(key, self.W_k)

        attention_scores = tf.matmul(q, k, transpose_b=True)
        attention_weights = tf.nn.softmax(attention_scores, axis=-1)

        output = tf.matmul(attention_weights, key)
        return output
