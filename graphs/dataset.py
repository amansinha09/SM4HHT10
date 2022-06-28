import dgl
from dgl.data import DGLDataset
import torch
import os
import pandas as pd
import numpy as np

class OurPubmedData(DGLDataset):
    def __init__(self):
        super().__init__(name='pubmed_data')
        self._verbose = True

    def process(self):
        # ======== fetching data from various files ========== #
        # load features
        featname = ['id', 'pubmedid'] + [f'f{i}' for i in range(500)]
        features = pd.read_csv('/home/amsinha/grm/GRM/notebooks/pubmedraw/features.dat', sep=' ', names = featname)
        features = features[features.pubmedid != 17874530].reset_index()
        features.index = list(range(len(features)))
        node_features = torch.from_numpy(features.iloc[:,3:].to_numpy(dtype=np.float32))

        pid2id = dict(zip(features.pubmedid, features.index))
        #print(pid2id.keys())

        # load labels
        labels = pd.read_csv('/home/amsinha/grm/GRM/notebooks/pubmedraw/labels.dat',sep=' ', names=['id', 'pubmedid', 'label'])
        labels = labels[labels.pubmedid != 17874530].reset_index()
        labels.id = list(range(len(labels)))
        node_labels = torch.from_numpy(labels.label.to_numpy())

        # load cites
        cites = pd.read_csv('/home/amsinha/grm/GRM/notebooks/pubmedraw/cites.dat', sep=' ', names= ['src_orig', 'dest_orig'])
        cites = cites[cites.dest_orig != 17874530].reset_index()
        cites['src'] = [pid2id[k] for k in cites.src_orig]
        cites['dest'] = [pid2id[k] for k in cites.dest_orig]
        src = cites.src.tolist() + cites.dest.tolist()
        dest = cites.dest.tolist() + cites.src.tolist()
        edges_src = torch.from_numpy(np.asarray(src))
        edges_dst = torch.from_numpy(np.asarray(dest))
        all_nodes = set(cites.src) | set(cites.dest)
        print(max(all_nodes), len(set(all_nodes)))

        # load masks
        masks = np.load('../masksplits/pubmedmask8.npy')
        train_mask = torch.from_numpy(masks[0])
        val_mask = torch.from_numpy(masks[1])
        test_mask = torch.from_numpy(masks[2])
        test_mask2 = torch.from_numpy(masks[3])
        
         #nodes_data = pd.read_csv('./members.csv')
        #edges_data = pd.read_csv('./interactions.csv')
        #node_features = torch.from_numpy(nodes_data['Age'].to_numpy())
        #node_labels = torch.from_numpy(nodes_data['Club'].astype('category').cat.codes.to_numpy())
        #edge_features = torch.from_numpy(edges_data['Weight'].to_numpy())
        #edges_src = torch.from_numpy(edges_data['Src'].to_numpy())
        #edges_dst = torch.from_numpy(edges_data['Dst'].to_numpy())
        #print(node_features.shape)
        self.graph = dgl.graph((edges_src, edges_dst), num_nodes=node_features.shape[0])
        self.graph.ndata['feat'] = node_features
        self.graph.ndata['label'] = node_labels
        #self.graph.edata['weight'] = edge_features
        self.graph.num_labels = len(set(node_labels))



        # If your dataset is a node classification dataset, you will need to assign
        # masks indicating whether a node belongs to training, validation, and test set.
        #n_nodes = nodes_data.shape[0]
        #n_train = int(n_nodes * 0.6)
        #n_val = int(n_nodes * 0.2)
        #train_mask = torch.zeros(n_nodes, dtype=torch.bool)
        #val_mask = torch.zeros(n_nodes, dtype=torch.bool)
        #test_mask = torch.zeros(n_nodes, dtype=torch.bool)
        #train_mask[:n_train] = True
        #val_mask[n_train:n_train + n_val] = True
        #test_mask[n_train + n_val:] = True
        self.graph.ndata['train_mask'] = train_mask
        self.graph.ndata['val_mask'] = val_mask
        self.graph.ndata['test_mask'] = test_mask
        self.graph.ndata['testt_mask'] = test_mask2

    def __getitem__(self, i):
        return self.graph
    
    @property
    def num_labels(self):
        """Number of classes."""
        return self.graph.num_labels

    def __len__(self):
        return 1

