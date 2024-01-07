from os import listdir
from os.path import isfile, join

import numpy as np
import pandas as pd

from src.dataset.instances.graph import GraphInstance
from src.dataset.generators.base import Generator

class ProteinsFull(Generator):
    
    def init(self):
        base_path = self.local_config['parameters']['data_dir']
        self._adj_file_path = join(base_path, 'PROTEINS_full_A.txt')
        self._ind_file_path = join(base_path, 'PROTEINS_full_graph_indicator.txt')
        self._lab_file_path = join(base_path, 'PROTEINS_full_graph_labels.txt')
        self._att_file_path = join(base_path, 'PROTEINS_full_node_attributes.txt')
        self._nlb_file_path = join(base_path, 'PROTEINS_full_node_labels.txt')

        self.dataset.node_features_map = {"is_lab_0":0, "is_lab_1":1, "is_lab_2":2}
        self.dataset.node_features_map.update({f"is_feat_{i}":i for i in range(3,32)})

        self.generate_dataset()
    
    def generate_dataset(self):
        if not len(self.dataset.instances):
            with open(self._ind_file_path, "rt") as handle:
                self.graph_indicator = np.asarray(list(map(int,filter(lambda x: len(x) > 0, handle.read().split("\n")))), dtype=np.uint64)
            with open(self._lab_file_path, "rt") as handle:
                self.all_graph_labels = np.asarray(list(map(int,filter(lambda x: len(x) > 0, handle.read().split("\n")))), dtype=np.uint8)
            with open(self._nlb_file_path, "rt") as handle:
                self.all_node_labels = np.asarray(list(map(int,filter(lambda x: len(x) > 0, handle.read().split("\n")))), dtype=np.uint8)

            self.all_node_attributes = pd.read_csv(self._att_file_path, header=None).values
            self.edges = pd.read_csv(self._adj_file_path, header=None).values

            # Create representations
            graphs = []
            adj_list = []
            graph_id = 1
            for edge in self.edges:
                if graph_id != self.graph_indicator[edge[0] - 1]:
                    graphs.append(self.create_graph(adj_list, graph_id))
                    # Prepare for another graph
                    adj_list = []
                    graph_id = self.graph_indicator[edge[0] - 1].item()
                adj_list.append(edge)

            # Create last graph!
            graphs.append(self.create_graph(adj_list, graph_id))

            one_hot_node_labels = np.eye(3,3)

            # Create instances
            instance_id = len(self.dataset.instances)
            for graph_label, mat, node_labels, node_attributes in graphs:
                node_labels = one_hot_node_labels[node_labels]
                node_features = np.concatenate([node_labels, node_attributes], axis=-1)
                self.dataset.instances.append(GraphInstance(instance_id, label=graph_label - 1, data=mat, node_features=node_features))
                instance_id += 1

    def create_graph(self, adj_list, graph_id):
        finder = [n for edge in adj_list for n in edge]
        min_node = min(finder)
        max_node = max(finder)

        # Create adj matrix, edges are undirected!
        mat = np.zeros((max_node - min_node + 1, max_node - min_node + 1), dtype=np.int32)
        for n1,n2 in adj_list:
            n1 -= min_node
            n2 -= min_node
            mat[n1, n2] = 1
            mat[n2, n1] = 1

        # Get graph label
        graph_label = self.all_graph_labels[graph_id - 1]

        # Get node labels
        node_labels = self.all_node_labels[min_node-1:max_node]

        # Get node attributes (should normalize?)
        node_attributes = self.all_node_attributes[min_node-1:max_node]

        return graph_label, mat, node_labels, node_attributes