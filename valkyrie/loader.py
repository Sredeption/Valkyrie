import networkx as nx
import pandas as pd

from valkyrie.dataset import HangzhouDataset


def load_hangzhou():
    g = nx.Graph()
    cites = pd.read_csv('./data/hangzhou/hangzhou.cites', sep="\t", header=None)
    content = pd.read_csv('./data/hangzhou/hangzhou.content', sep="\t", header=None).to_numpy()

    for index, edge in cites.iterrows():
        g.add_edge(int(edge[0]), int(edge[1]))

    return HangzhouDataset(g, content[:, :-2], content[:, -1])


if __name__ == '__main__':
    pass
