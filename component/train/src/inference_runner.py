from __future__ import (division, print_function)

import json
import os
import numpy as np
from tqdm import tqdm

import torch
import torch.utils.data
import torch.optim as optim
from torch.utils.tensorboard.writer import SummaryWriter
from model.gnn import NodeGNN
from dataset.dataloader import *
import bentoml

EPS = float(np.finfo(np.float32).eps)
__all__ = ['NeuralInferenceRunner']

class NeuralInferenceRunner(object):

  def train(self, train_data, val_data):
    print("=== START TRAINING ===")
    print("USING GPU" if torch.cuda.is_available() else "USING CPU")
    # create data loader
    train_dataset = MyDataloader(train_data)
    val_dataset = MyDataloader(val_data)

    train_loader = torch.utils.data.DataLoader(
      train_dataset,
      batch_size=10,
      shuffle=True,
      collate_fn=train_dataset.collate_fn)

    val_loader = torch.utils.data.DataLoader(
      val_dataset,
      batch_size=10,
      shuffle=True,
      collate_fn=val_dataset.collate_fn)

    # create models
    model = NodeGNN()

    # create optimizer
    params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.Adam(
        params,
        lr=0.001)

    # reset gradient
    optimizer.zero_grad()

    # writer
    tb_log_dir = './log'
    writer = SummaryWriter(tb_log_dir)
    iter_count_train = 0
    iter_count_val = 0

    #========================= Training Loop =============================#
    best_val_loss = np.inf
    best_train_loss = np.inf
    for epoch in range(10):
      print("=== EPOCH : {} ===".format(epoch))
      # ===================== validation ============================ #
      model.eval()
      val_loss_avg = []
      for data in tqdm(val_loader, desc="VALIDATION"):
        with torch.no_grad():
          _, val_loss = model(data['J_msg'], data['b'], data['msg_node'], target=data['prob_gt'])
          val_loss_avg.append(val_loss.detach().numpy())
        writer.add_scalar('val_loss', val_loss, iter_count_val)
        iter_count_val += 1

      if best_val_loss > np.mean(val_loss_avg):
        best_val_loss = np.mean(val_loss_avg)

      # ====================== training ============================= #
      model.train()
      train_loss_avg = []
      for data in tqdm(train_loader, desc="TRAINING"):
        optimizer.zero_grad()
        _, train_loss = model(data['J_msg'], data['b'], data['msg_node'],  target=data['prob_gt'])
        train_loss_avg.append(train_loss.detach().numpy())
        train_loss.backward()
        optimizer.step()

        writer.add_scalar('train_loss', train_loss, iter_count_train)
        iter_count_train += 1

      if best_train_loss > np.mean(train_loss_avg):
        best_train_loss = np.mean(train_loss_avg)


    print("===================")
    print("TRAINING FINISHED")
    print("BEST VAL LOSS: {}".format(best_val_loss))
    print("BEST BEST LOSS: {}".format(best_train_loss))
    print("===================")
    writer.close()
    snapshot(model, optimizer)
    bentoml.pytorch.save('simple_gnn', model)

    bentoml.build(
      "service.py:svc",
      include=["*.py"],
      python=dict(
        packages=["torch"]
      )
    )

    bentoml.export_bento('simple_gnn:latest', './data/simple_gnn.bento')

    metadata = {
      'outputs': [{
        'type': 'tensorboard',
        'source': tb_log_dir
      }]
      }

    with open('./data/mlpipeline-ui-metadata.json', 'w') as f:
      json.dump(metadata, f)

    metrics = {
      'metrics': [
        {
          'name': 'best_val_loss',
          'numberValue': float(best_val_loss),
          'format': 'RAW'
        },
        {
          'name': 'best_train_loss',
          'numberValue': float(best_train_loss),
          'format': 'RAW'
        }
      ]
    }

    with open('./data/mlpipeline-metrics.json', 'w') as f:
      json.dump(metrics, f)

    return best_val_loss


def snapshot(model, optimizer):
  model_snapshot = {
    "model": model.state_dict(),
    "optimizer": optimizer.state_dict(),
  }
  torch.save(model_snapshot,
             os.path.join("data", "model_snapshot.pth"))