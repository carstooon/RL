import tensorflow as tf

def OneLayerNetwork(input, output, neurons):
    fc1 = tf.contrib.layers.fully_connected(input, num_outputs=neurons, activation_fn=tf.nn.relu,
                                            weights_initializer=tf.contrib.layers.xavier_initializer(),
                                            biases_initializer=tf.constant_initializer(0.1))
    out_layer = tf.contrib.layers.fully_connected(fc1, num_outputs=output, activation_fn=None,
                                            weights_initializer=tf.contrib.layers.xavier_initializer(),
                                            biases_initializer=tf.constant_initializer(0.1))
    return out_layer

def TwoLayerNetwork(input, output, neurons1, neurons2):

    fc1 = tf.contrib.layers.fully_connected(input, num_outputs=neurons1, activation_fn=tf.nn.relu,
                                            weights_initializer=tf.contrib.layers.xavier_initializer(),
                                            biases_initializer=tf.constant_initializer(0.1))
    fc2 = tf.contrib.layers.fully_connected(fc1, num_outputs=neurons2, activation_fn=tf.nn.relu,
                                            weights_initializer=tf.contrib.layers.xavier_initializer(),
                                            biases_initializer=tf.constant_initializer(0.1))
    out_layer = tf.contrib.layers.fully_connected(fc1, num_outputs=output, activation_fn=None,
                                            weights_initializer=tf.contrib.layers.xavier_initializer(),
                                            biases_initializer=tf.constant_initializer(0.1))
    return out_layer
