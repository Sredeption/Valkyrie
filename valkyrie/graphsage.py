from collections import namedtuple

import tensorflow as tf
from tensorflow.keras import layers

from valkyrie.aggregators import MeanAggregator
from valkyrie.neigh_samplers import UniformNeighborSampler

SAGEInfo = namedtuple("SAGEInfo",
                      ['layer_name',  # name of the layer (to get feature embedding etc.)
                       'num_samples',
                       'output_dim'  # the output (i.e., hidden) dimension
                       ])


class Sample(layers.Layer):
    def __init__(self, adj_info, layer_infos):
        super(Sample, self).__init__()
        self.adj_info = adj_info
        self.layer_infos = layer_infos
        self.num_samplers = len(self.layer_infos)
        self.neighbor_samplers = [
            UniformNeighborSampler(self.adj_info, k, layer_infos[self.num_samplers - k - 1].num_samples)
            for k in range(self.num_samplers)]

    def call(self, inputs, **kwargs):
        samples = [inputs]
        for k, sampler in enumerate(self.neighbor_samplers):
            samples.append(sampler(samples[k]))
        return samples


class Aggregate(layers.Layer):
    def __init__(self, features, layer_infos):
        super(Aggregate, self).__init__()
        self.features = features
        self.depth = len(layer_infos)
        self.aggregators = [MeanAggregator(layer_infos[k].output_dim, concat=True) for k in range(self.depth)]

    def call(self, inputs, **kwargs):
        hidden = [tf.nn.embedding_lookup(self.features, node_samples) for node_samples in inputs]
        for k in range(self.depth):
            next_hidden = []
            for hop in range(self.depth - k):
                h = self.aggregators[k]((hidden[hop], hidden[hop + 1]))
                next_hidden.append(h)
            hidden = next_hidden
        return hidden[0]


class SampleAndAggregate(tf.keras.Model):
    def __init__(self, features, adj_info, layer_infos):
        super(SampleAndAggregate, self).__init__()
        self.features = tf.Variable(tf.constant(features, dtype=tf.float32), trainable=False, name="features")
        self.adj_info = tf.Variable(tf.constant(adj_info, dtype=tf.int32), trainable=False, name="adj_info")
        self.sample_layer = Sample(self.adj_info, layer_infos)
        self.aggregate_layer = Aggregate(self.features, layer_infos)
        self.output_layer = layers.Dense(1)

    def call(self, inputs, training=None, mask=None):
        node_indices = tf.reshape(inputs, [-1])
        samples = self.sample_layer(node_indices)
        outputs = self.aggregate_layer(samples)
        outputs = tf.nn.l2_normalize(outputs, 1)
        outputs = self.output_layer(outputs)
        return outputs
