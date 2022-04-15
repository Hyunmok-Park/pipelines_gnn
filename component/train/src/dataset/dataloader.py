import os
import glob
import torch
import pickle
import numpy as np

from torch.utils.data import Dataset

__all__ = ['MyDataloader']

class MyDataloader(Dataset):
    def __init__(self, data_file):
        self.data = pickle.load(open(data_file, "rb"))
        self.gts = self.data['prob_gt']
        self.bs = self.data['b']
        self.Js = self.data['J_msg']
        self.msg_nodes = self.data['msg_node']

        self.num_graphs = self.gts.shape[0]

    def __getitem__(self, index):
        graph = {}

        graph['prob_gt'] = self.gts[index]
        graph['J_msg'] = self.Js[index]
        graph['b'] = self.bs[index]
        graph['msg_node'] = self.msg_nodes[index]

        return graph

    def __len__(self):
        return self.num_graphs

    def collate_fn(self, batch): # batch : list of dicts
        assert isinstance(batch, list)
        data = {}

        data['prob_gt'] = torch.from_numpy(
            np.concatenate([bch['prob_gt'] for bch in batch], axis=0)).float()
        data['J_msg'] = torch.from_numpy(
            np.concatenate([bch['J_msg'] for bch in batch], axis=0)).float()
        data['b'] = torch.from_numpy(
            np.concatenate([bch['b'] for bch in batch], axis=0)).float()

        msg_node = np.empty((0, 2))
        num_msg_node = 0
        for bch in batch:
            msg_node = np.vstack((msg_node, num_msg_node + bch['msg_node']))
            num_msg_node = 1 + msg_node.max()
        data['msg_node'] = torch.from_numpy(msg_node).long()
        return data