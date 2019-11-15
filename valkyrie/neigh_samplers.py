import tensorflow as tf
from tensorflow.keras import layers


class UniformNeighborSampler(layers.Layer):
    """
    Uniformly samples neighbors.
    Assumes that adj lists are padded with random re-sampling
    """

    def __init__(self, adj_info, k, num_samples):
        super(UniformNeighborSampler, self).__init__()
        self.adj_info = adj_info
        self.k = k
        self.begins = []
        self.sizes = []
        for i in range(k + 1):
            self.begins.append(0)
            self.sizes.append(-1)
        self.begins.append(0)
        self.sizes.append(num_samples)

    def call(self, inputs, **kwargs):
        adj_lists = tf.nn.embedding_lookup(self.adj_info, inputs)
        adj_lists = tf.transpose(tf.random.shuffle(tf.transpose(adj_lists)))

        adj_lists = tf.slice(adj_lists, begin=self.begins, size=self.sizes)

        return adj_lists
