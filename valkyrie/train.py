import os

import numpy as np
import tensorflow as tf
from absl import app
from absl import flags

from valkyrie import loader
from valkyrie.graphsage import SampleAndAggregate, SAGEInfo
from tensorflow.keras import metrics

seed = 123
np.random.seed(seed)
tf.random.set_seed(seed)

# Settings
FLAGS = flags.FLAGS

flags.DEFINE_boolean('log_device_placement', False,
                     """Whether to log device placement.""")
# core params..
flags.DEFINE_string('model', 'graphsage_mean', 'model names. See README for possible values.')
flags.DEFINE_float('learning_rate', 0.01, 'initial learning rate.')
flags.DEFINE_string("model_size", "small", "Can be big or small; model specific def'ns")
flags.DEFINE_string('train_prefix', '', 'prefix identifying training data. must be specified.')

# left to default values in main experiments
flags.DEFINE_integer('epochs', 10, 'number of epochs to train.')
flags.DEFINE_float('dropout', 0.0, 'dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_decay', 0.0, 'weight for l2 loss on embedding matrix.')
flags.DEFINE_integer('max_degree', 128, 'maximum node degree.')
flags.DEFINE_integer('samples_1', 25, 'number of samples in layer 1')
flags.DEFINE_integer('samples_2', 10, 'number of samples in layer 2')
flags.DEFINE_integer('samples_3', 0, 'number of users samples in layer 3. (Only for mean model)')
flags.DEFINE_integer('dim_1', 128, 'Size of output dim (final is 2x this, if using concat)')
flags.DEFINE_integer('dim_2', 128, 'Size of output dim (final is 2x this, if using concat)')
flags.DEFINE_boolean('random_context', True, 'Whether to use random context or direct edges')
flags.DEFINE_integer('batch_size', 512, 'minibatch size.')
flags.DEFINE_boolean('sigmoid', False, 'whether to use sigmoid loss')
flags.DEFINE_integer('identity_dim', 0,
                     'Set to positive value to use identity embedding features of that dimension. Default 0.')

# logging, saving, validation settings etc.
flags.DEFINE_string('base_log_dir', '.', 'base directory for logging and saving embeddings')
flags.DEFINE_integer('validate_iter', 5000, "how often to run a validation minibatch.")
flags.DEFINE_integer('validate_batch_size', 256, "how many nodes per validation sample.")
flags.DEFINE_integer('gpu', 0, "which gpu to use.")
flags.DEFINE_integer('print_every', 5, "How often to print training info.")
flags.DEFINE_integer('max_total_steps', 10 ** 10, "Maximum total number of iterations")

GPU_MEM_FRACTION = 0.8


def main(argv=None):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(FLAGS.gpu)
    print("Loading training data..")
    data = loader.load_hangzhou()
    dataset = data.train_dataset
    print("Done loading training data..")

    if FLAGS.samples_3 != 0:
        layer_infos = [SAGEInfo("node", FLAGS.samples_1, FLAGS.dim_1),
                       SAGEInfo("node", FLAGS.samples_2, FLAGS.dim_2),
                       SAGEInfo("node", FLAGS.samples_3, FLAGS.dim_2)]
    elif FLAGS.samples_2 != 0:
        layer_infos = [SAGEInfo("node", FLAGS.samples_1, FLAGS.dim_1),
                       SAGEInfo("node", FLAGS.samples_2, FLAGS.dim_2)]
    else:
        layer_infos = [SAGEInfo("node", FLAGS.samples_1, FLAGS.dim_1)]

    sage = SampleAndAggregate(data.features, data.adj, layer_infos)

    optimizer = tf.keras.optimizers.Adam(learning_rate=FLAGS.learning_rate)
    loss = tf.keras.losses.MSE

    sage.compile(optimizer=optimizer, loss=loss, metrics=[metrics.MSE])

    for epoch in range(FLAGS.epochs):
        for x, y in dataset:
            sage.fit(x, y, verbose=1)


if __name__ == '__main__':
    app.run(main)
