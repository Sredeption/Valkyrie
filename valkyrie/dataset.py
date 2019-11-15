import numpy as np
import tensorflow as tf


class HangzhouDataset:

    def __init__(self, g, features, labels, batch_size=128, max_degree=25):
        self.g = g
        self.num_node = len(self.g.nodes)
        self.max_degree = max_degree
        self.adj, self.deg = self.construct_adj()
        self.features = features

        nodes = []
        targets = []
        for node in g.nodes():
            node_id = int(node)
            nodes.append(node_id)
            targets.append(labels[node_id])

        self.train_dataset = tf.data.Dataset.from_tensor_slices((nodes, targets)).batch(batch_size)

    def construct_adj(self):
        adj = self.num_node * np.ones((self.num_node, self.max_degree))
        deg = np.zeros((self.num_node,))

        for node in self.g.nodes():
            node_id = int(node)
            neighbors = np.array([int(neighbor) for neighbor in self.g.neighbors(node)])
            deg[node_id] = len(neighbors)
            if len(neighbors) == 0:
                raise AttributeError()
            neighbors = np.random.choice(neighbors, self.max_degree, replace=len(neighbors) < self.max_degree)
            adj[node_id, :] = neighbors
        return adj, deg
