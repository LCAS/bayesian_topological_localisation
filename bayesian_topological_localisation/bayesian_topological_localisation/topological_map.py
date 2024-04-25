#!/usr/bin/env python3

import numpy as np


class TopologicalMap():

  def __init__(self, msg=None):
    if msg is not None:
      # save and compute topological map informations
      self.node_names = np.array([node.name for node in msg.nodes])
      self.node_coords = np.array(
        [[node.pose.position.x, node.pose.position.y] for node in msg.nodes])

      self.node_diffs2D = []
      self.node_distances = []
      self.connected_nodes = []
      for i, _ in enumerate(self.node_names):
        self.node_diffs2D.append(self.node_coords - self.node_coords[i])
        self.connected_nodes.append(
            np.array([np.where(self.node_names == edge.node)[0][0] for edge in msg.nodes[i].edges]))

      self.node_diffs2D = np.array(self.node_diffs2D)
      
      self.node_distances = np.sqrt(np.sum(self.node_diffs2D ** 2, axis=2))

    else:
      self.node_diffs2D = []
      self.node_distances = []
      self.connected_nodes = []
      self.node_names = []
      self.node_coords = np.array([])
