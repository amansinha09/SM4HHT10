import dgl
from dgl.data import DGLDataset
import torch
import os
import pandas as pd
import numpy as np

class MedNER(DGLDataset):
    def __init__(self):
        super().__init__(name='medner')
        self._verbose = True

    def process(self):
        # ======== fetching data from various files ========== #
        # load features
        features = np.load('/home/amansinha/SM4HHT10/graphs/data/random_features.npz')['feat']
        node_features = torch.from_numpy(features)

        # load labels
        labels = pd.read_csv('/home/amansinha/SM4HHT10/graphs/data/conll.labels',sep='\t', names=['id', 'label'])
        label_encoder = {' O':0, ' B-MISC':1, ' I-MISC': 2}
        labels.label = [label_encoder[l] for l in labels.label]
        labels.id = list(range(len(labels)))
        node_labels = torch.from_numpy(labels.label.to_numpy())

        # load cites
        cites = pd.read_csv('/home/amansinha/SM4HHT10/graphs/data/context.edges', sep='\t', names= ['src', 'dest', 'rel'] )
        src = cites.src.tolist()
        dest = cites.dest.tolist()
        edges_src = torch.from_numpy(np.asarray(src))
        edges_dst = torch.from_numpy(np.asarray(dest))
        all_nodes = set(cites.src) | set(cites.dest)
        print(max(all_nodes), len(set(all_nodes)))

        # load masks
        masks = np.load('./../graphs/data/masks.npz')
        train_mask = torch.from_numpy(masks['train'])
        val_mask = torch.from_numpy(masks['val'])
        test_mask = torch.from_numpy(masks['test'])
        

        self.graph = dgl.graph((edges_src, edges_dst), num_nodes=node_features.shape[0])
        self.graph.ndata['feat'] = node_features
        self.graph.ndata['label'] = node_labels
        self.graph.num_labels = len(set(node_labels))


        self.graph.ndata['train_mask'] = train_mask
        self.graph.ndata['val_mask'] = val_mask
        self.graph.ndata['test_mask'] = test_mask

    def __getitem__(self, i):
        return self.graph
    
    @property
    def num_labels(self):
        """Number of classes."""
        return self.graph.num_labels
    
    @property
    def num_classes(self):
        """Number of classes."""
        return self.graph.num_labels

    def __len__(self):
        return 1
