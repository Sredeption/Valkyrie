from abc import abstractmethod, ABCMeta

import tensorflow as tf
from tensorflow.keras import layers


class SageAggregator(layers.Layer, metaclass=ABCMeta):

    def __init__(self, unit, bias, act, concat):
        super(SageAggregator, self).__init__()

        self.unit = unit
        self.bias = bias
        self.act = act
        self.concat = concat
        self.neigh_weights = None
        self.self_weights = None
        self.bias_weight = None

    def build(self, input_shape):
        self_shape, neigh_shape = input_shape
        self.neigh_weights = self.add_weight(
            shape=(neigh_shape[-1], self.unit),
            initializer='glorot_uniform',
            trainable=True)

        self.self_weights = self.add_weight(
            shape=(self_shape[-1], self.unit),
            initializer='glorot_uniform',
            trainable=True)
        if self.bias:
            self.bias_weight = self.add_weight(
                shape=(self.unit,),
                initializer='zeros',
                trainable=True)

    def _neighbor_model(self, neigh_vecs):
        raise NotImplementedError()

    def _self_model(self, self_vecs):
        raise NotImplementedError()

    def call(self, inputs, **kwargs):
        self_vecs, neigh_vecs = inputs
        neighs_logit = tf.matmul(self._neighbor_model(neigh_vecs), self.neigh_weights)
        self_logit = tf.matmul(self._self_model(self_vecs), self.self_weights)

        if not self.concat:
            output = tf.add_n([self_logit, neighs_logit])
        else:
            output = tf.concat([self_logit, neighs_logit], axis=-1)

        if self.bias:
            output += self.bias_weight

        return self.act(output)


class MeanAggregator(SageAggregator):
    """
    Aggregates via mean followed by matmul and non-linearity.
    """

    def __init__(self, unit, dropout=0., bias=False, act=tf.nn.relu,
                 concat=False):
        super(MeanAggregator, self).__init__(unit, bias, act, concat)

        self.dropout = dropout

    def _neighbor_model(self, neigh_vecs):
        neigh_vecs = tf.nn.dropout(neigh_vecs, self.dropout)
        return tf.reduce_mean(neigh_vecs, axis=-2)

    def _self_model(self, self_vecs):
        return tf.nn.dropout(self_vecs, self.dropout)


class GCNAggregator(layers.Layer):
    """
    Aggregates via mean followed by matmul and non-linearity.
    Same matmul parameters are used self vector and neighbor vectors.
    """

    def __init__(self, input_dim, output_dim, neigh_input_dim=None, dropout=0., bias=False, act=tf.nn.relu,
                 concat=False):
        super(GCNAggregator, self).__init__()

        self.dropout = dropout
        self.bias = bias
        self.act = act
        self.concat = concat

        if neigh_input_dim is None:
            neigh_input_dim = input_dim

        self.neigh_weights = self.add_weight(
            shape=(neigh_input_dim, output_dim),
            initializer='glorot_uniform',
            trainable=True)

        if self.bias:
            self.bias_weight = self.add_weight(
                shape=(output_dim,),
                initializer='zeros',
                trainable=True)

    def call(self, inputs, **kwargs):
        self_vecs, neigh_vecs = inputs

        neigh_vecs = tf.nn.dropout(neigh_vecs, self.dropout)
        self_vecs = tf.nn.dropout(self_vecs, self.dropout)
        means = tf.reduce_mean(tf.concat([neigh_vecs,
                                          tf.expand_dims(self_vecs, axis=1)], axis=1), axis=1)

        # [nodes] x [out_dim]
        output = tf.matmul(means, self.neigh_weights)

        # bias
        if self.bias:
            output += self.bias_weight

        return self.act(output)


class PoolingAggregator(SageAggregator, metaclass=ABCMeta):

    def __init__(self, input_dim, output_dim, neigh_feature_dim, bias, act, concat):
        super(PoolingAggregator, self).__init__(input_dim, output_dim, neigh_feature_dim, bias, act, concat)

    def _neighbor_model(self, neigh_vecs):
        neigh_h = neigh_vecs

        dims = tf.shape(neigh_h)
        batch_size = dims[0]
        num_neighbors = dims[1]
        h_reshaped = tf.reshape(neigh_h, (batch_size * num_neighbors, self.neigh_input_dim))

        for l in self._mlp_layers():
            h_reshaped = l(h_reshaped)
        neigh_h = tf.reshape(h_reshaped, (batch_size, num_neighbors, self.hidden_dim))
        return self.pooling(neigh_h)

    def _self_model(self, self_vecs):
        return self_vecs

    @abstractmethod
    def _pooling(self, neigh_h):
        raise NotImplementedError()

    @abstractmethod
    def _mlp_layers(self):
        raise NotImplementedError()


class MaxPoolingAggregator(PoolingAggregator):
    """
    Aggregates via max-pooling over MLP functions.
    """

    def __init__(self, input_dim, output_dim, model_size="small",
                 dropout=0., bias=False, act=tf.nn.relu, concat=False):
        if model_size == "small":
            hidden_dim = self.hidden_dim = 512
        elif model_size == "big":
            hidden_dim = self.hidden_dim = 1024
        else:
            raise AttributeError("model_size should be small or big")
        super(MaxPoolingAggregator, self).__init__(
            input_dim, output_dim, hidden_dim, bias, act, concat
        )

        self.mlp_layers = [
            layers.Dense(hidden_dim, activation="relu"),
            layers.Dropout(dropout)
        ]

    def _pooling(self, neigh_h):
        return tf.reduce_max(neigh_h, axis=1)

    def _mlp_layers(self):
        return self.mlp_layers


class MeanPoolingAggregator(PoolingAggregator):
    """ Aggregates via mean-pooling over MLP functions.
    """

    def __init__(self, input_dim, output_dim, model_size="small", dropout=0., bias=False, act=tf.nn.relu, concat=False):
        if model_size == "small":
            hidden_dim = self.hidden_dim = 512
        elif model_size == "big":
            hidden_dim = self.hidden_dim = 1024
        else:
            raise AttributeError("model_size should be small or big")
        super(MeanPoolingAggregator, self).__init__(
            input_dim, output_dim, hidden_dim, bias, act, concat
        )

        self.mlp_layers = [
            layers.Dense(hidden_dim, activation="relu"),
            layers.Dropout(dropout)
        ]

    def _pooling(self, neigh_h):
        return tf.reduce_mean(neigh_h, axis=1)

    def _mlp_layers(self):
        return self.mlp_layers


class TwoMaxLayerPoolingAggregator(PoolingAggregator):
    """ Aggregates via pooling over two MLP functions.
    """

    def __init__(self, input_dim, output_dim, model_size="small", dropout=0., bias=False, act=tf.nn.relu, concat=False):
        if model_size == "small":
            hidden_dim_1 = self.hidden_dim_1 = 512
            hidden_dim_2 = self.hidden_dim_2 = 256
        elif model_size == "big":
            hidden_dim_1 = self.hidden_dim_1 = 1024
            hidden_dim_2 = self.hidden_dim_2 = 512
        else:
            raise AttributeError("model_size should be small or big")
        super(TwoMaxLayerPoolingAggregator, self).__init__(
            input_dim, output_dim, hidden_dim_2, bias, act, concat
        )

        self.mlp_layers = [
            layers.Dense(hidden_dim_1, activation="relu"),
            layers.Dropout(dropout),
            layers.Dense(hidden_dim_2, activation="relu"),
            layers.Dropout(dropout),
        ]

    def _pooling(self, neigh_h):
        return tf.reduce_max(neigh_h, axis=1)

    def _mlp_layers(self):
        return self.mlp_layers


class SeqAggregator(SageAggregator):
    """ Aggregates via a standard LSTM.
    """

    def __init__(self, input_dim, output_dim, model_size="small", bias=False, act=tf.nn.relu, concat=False):
        if model_size == "small":
            hidden_dim = self.hidden_dim = 128
        elif model_size == "big":
            hidden_dim = self.hidden_dim = 256
        else:
            raise AttributeError("model_size should be small or big")
        super(SeqAggregator, self).__init__(input_dim, output_dim, hidden_dim, bias, act, concat)

        self.cell = layers.LSTMCell(self.hidden_dim)

    def _neighbor_model(self, neigh_vecs):
        dims = tf.shape(neigh_vecs)
        batch_size = dims[0]
        initial_state = self.cell.zero_state(batch_size, tf.float32)
        used = tf.sign(tf.reduce_max(tf.abs(neigh_vecs), axis=2))
        length = tf.reduce_sum(used, axis=1)
        length = tf.maximum(length, tf.constant(1.))
        length = tf.cast(length, tf.int32)

        rnn_outputs = layers.RNN(self.cell, time_major=False)(neigh_vecs, initial_state=initial_state)

        batch_size = tf.shape(rnn_outputs)[0]
        max_len = tf.shape(rnn_outputs)[1]
        out_size = int(rnn_outputs.get_shape()[2])
        index = tf.range(0, batch_size) * max_len + (length - 1)
        flat = tf.reshape(rnn_outputs, [-1, out_size])
        return tf.gather(flat, index)

    def _self_model(self, self_vecs):
        return self_vecs
