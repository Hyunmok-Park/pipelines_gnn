import os
import pickle
import numpy as np
from topology import NetworkTopology, get_msg_graph
from gt_inference import Enumerate
from scipy.sparse import coo_matrix
import shutil

def mkdir(dir_name):
  if not os.path.exists(dir_name):
    os.makedirs(dir_name)
    print('made directory {}'.format(dir_name))

def main():

    num_nodes_I = 9
    std_J_A = 1/3
    std_b_A = 0.25

    train_gt = []
    train_b = []
    train_J = []
    train_msg_node = []

    for sample_id in range(100):
        seed_train = int(str(1111) + str(sample_id))#
        topology = NetworkTopology(num_nodes=num_nodes_I, seed=seed_train)#
        npr = np.random.RandomState(seed=seed_train)#
        G, W = topology.generate(topology='grid')
        J = npr.normal(0, std_J_A, size=[num_nodes_I, num_nodes_I])
        J = (J + J.transpose()) / 2.0
        J = J * W

        b = npr.normal(0, std_b_A, size=[num_nodes_I, 1])

        # Enumerate
        model = Enumerate(W, J, b)
        prob_gt = model.inference()

        msg_node, msg_adj = get_msg_graph(G)
        msg_node, msg_adj = np.array(msg_node), np.array(msg_adj)
        J_msg = J[msg_node[:, 0], msg_node[:, 1]].reshape(-1, 1)

        train_gt.append([prob_gt])
        train_b.append([b])
        train_J.append([J_msg])
        train_msg_node.append([msg_node])

    train_gt = np.concatenate(train_gt, axis=0)
    train_b = np.concatenate(train_b, axis=0)
    train_J = np.concatenate(train_J, axis=0)
    train_msg_node = np.concatenate(train_msg_node, axis=0)

    # np.save("train_gt.npy", train_gt)
    # np.save("train_b.npy", train_b)
    # np.save("train_J.npy", train_J)
    # np.save("train_msg_node.npy", train_msg_node)

    train_graphs = {}
    train_graphs['prob_gt'] = train_gt
    train_graphs['b'] = train_b
    train_graphs['J_msg'] = train_J
    train_graphs['msg_node'] = train_msg_node

    with open("./train_data.p", "wb") as f:
        pickle.dump(train_graphs, f)


    num_nodes_I = 9
    std_J_A = 1/3
    std_b_A = 0.25

    val_gt = []
    val_b = []
    val_J = []
    val_msg_node = []

    for sample_id in range(10):
        seed_val = int(str(2222) + str(sample_id))#
        topology = NetworkTopology(num_nodes=num_nodes_I, seed=seed_val)#
        npr = np.random.RandomState(seed=seed_val)#
        G, W = topology.generate(topology='grid')
        J = npr.normal(0, std_J_A, size=[num_nodes_I, num_nodes_I])
        J = (J + J.transpose()) / 2.0
        J = J * W

        b = npr.normal(0, std_b_A, size=[num_nodes_I, 1])

        # Enumerate
        model = Enumerate(W, J, b)
        prob_gt = model.inference()

        msg_node, msg_adj = get_msg_graph(G)
        msg_node, msg_adj = np.array(msg_node), np.array(msg_adj)
        J_msg = J[msg_node[:, 0], msg_node[:, 1]].reshape(-1, 1)

        val_gt.append([prob_gt])
        val_b.append([b])
        val_J.append([J_msg])
        val_msg_node.append([msg_node])

    val_gt = np.concatenate(val_gt, axis=0)
    val_b = np.concatenate(val_b, axis=0)
    val_J = np.concatenate(val_J, axis=0)
    val_msg_node = np.concatenate(val_msg_node, axis=0)

    # np.save("val_gt.npy", val_gt)
    # np.save("val_b.npy", val_b)
    # np.save("val_J.npy", val_J)
    # np.save("val_msg_node.npy", val_msg_node)

    val_graphs = {}
    val_graphs['prob_gt'] = val_gt
    val_graphs['b'] = val_b
    val_graphs['J_msg'] = val_J
    val_graphs['msg_node'] = val_msg_node

    with open("./val_data.p", "wb") as f:
        pickle.dump(val_graphs, f)

    print(os.listdir("./"))

if __name__ == '__main__':
    main()