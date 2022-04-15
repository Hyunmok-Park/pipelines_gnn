import numpy as np
import torch
import torch.nn as nn

EPS = float(np.finfo(np.float32).eps)
__all__ = ['NodeGNN']

class NodeGNN(nn.Module):
  def __init__(self):
    """ A simplified implementation of NodeGNN """
    super(NodeGNN, self).__init__()
    self.hidden_dim = 16
    self.num_prop = 5
    self.aggregate_type = 'sum'

    # message function
    self.msg_func = nn.Sequential(*[
      nn.Linear(2 * self.hidden_dim + 8, 64),
      nn.ReLU(),
      nn.Linear(64, self.hidden_dim)
    ])

    # update function
    self.update_func = nn.GRUCell(
      input_size=self.hidden_dim, hidden_size=self.hidden_dim)

    # output function
    self.output_func = nn.Sequential(*[
      nn.Linear(self.hidden_dim + 2, 64),
      nn.ReLU(),
      nn.Linear(64, 2),
    ])
    self.loss_func = nn.KLDivLoss(reduction='batchmean')

  def forward(self, J_msg, b, msg_node, target=None):
    num_node = b.shape[0]
    num_edge = msg_node.shape[0]

    edge_in = msg_node[:, 0]
    edge_out = msg_node[:, 1].contiguous()

    ff_in = torch.cat([b[edge_in], -b[edge_in], J_msg, -J_msg], dim=1)
    ff_out = torch.cat([-b[edge_out], b[edge_out], -J_msg, J_msg], dim=1)

    state = torch.zeros(num_node, self.hidden_dim).to(b.device)

    def _prop(state_prev):
      # 1. compute messages
      state_in = state_prev[edge_in, :]  # shape |E| X D
      state_out = state_prev[edge_out, :]  # shape |E| X D
      msg = self.msg_func(torch.cat([state_in, ff_in, state_out, ff_out], dim=1)) # shape: |E| X D
      # 2. aggregate message
      scatter_idx = edge_out.view(-1, 1).expand(-1, self.hidden_dim)
      msg_agg = torch.zeros(num_node, self.hidden_dim).to(b.device) # shape: |V| X D
      msg_agg = msg_agg.scatter_add(0, scatter_idx, msg)
      avg_norm = torch.zeros(num_node).to(b.device).scatter_add_(0, edge_out, torch.ones(num_edge).to(b.device))
      msg_agg /= (avg_norm.view(-1, 1) + EPS)
      # 3. update state
      state_new = self.update_func(msg_agg, state_prev)  # GRU update
      return state_new

    # propagation
    for tt in range(self.num_prop):
      state = _prop(state)

    # output
    y = self.output_func(torch.cat([state, b, -b], dim=1))
    y = torch.log_softmax(y, dim=1)
    loss = self.loss_func(y, target)
    return y, loss

  def predict(self, J_msg, b, msg_node, prob_gt):
    J_msg = J_msg[0].long()
    b = b[0].long()
    msg_node = msg_node[0].long()

    num_node = b.shape[0]
    num_edge = msg_node.shape[0]

    edge_in = msg_node[:, 0]
    edge_out = msg_node[:, 1].contiguous()

    ff_in = torch.cat([b[edge_in], -b[edge_in], J_msg, -J_msg], dim=1)
    ff_out = torch.cat([-b[edge_out], b[edge_out], -J_msg, J_msg], dim=1)

    state = torch.zeros(num_node, self.hidden_dim).to(b.device)

    def _prop(state_prev):
      # 1. compute messages
      state_in = state_prev[edge_in, :]  # shape |E| X D
      state_out = state_prev[edge_out, :]  # shape |E| X D
      msg = self.msg_func(torch.cat([state_in, ff_in, state_out, ff_out], dim=1)) # shape: |E| X D
      # 2. aggregate message
      scatter_idx = edge_out.view(-1, 1).expand(-1, self.hidden_dim)
      msg_agg = torch.zeros(num_node, self.hidden_dim).to(b.device) # shape: |V| X D
      msg_agg = msg_agg.scatter_add(0, scatter_idx, msg)
      avg_norm = torch.zeros(num_node).to(b.device).scatter_add_(0, edge_out, torch.ones(num_edge).to(b.device))
      msg_agg /= (avg_norm.view(-1, 1) + EPS)
      # 3. update state
      state_new = self.update_func(msg_agg, state_prev)  # GRU update
      return state_new

    # propagation
    for tt in range(self.num_prop):
      state = _prop(state)

    # output
    res = dict()
    y = self.output_func(torch.cat([state, b, -b], dim=1))
    y = torch.log_softmax(y, dim=1)
    loss = self.loss_func(y, prob_gt)
    res["prob"] = np.exp(y.detach().cpu().numpy())
    res["loss"] = loss.detach().cpu().numpy()
    return [res]