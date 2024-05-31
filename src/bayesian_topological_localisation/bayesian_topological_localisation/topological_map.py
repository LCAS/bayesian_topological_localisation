#!/usr/bin/env python3

import numpy as np
import json

#class TopologicalMap():
#
#  def __init__(self, msg=None):
#    if msg is not None:
#      # save and compute topological map informations
#      self.node_names = np.array([node.name for node in msg.nodes])
#      self.node_coords = np.array([[node.pose.position.x, node.pose.position.y] for node in msg.nodes], dtype=float)
#
#      self.node_diffs2D = []
#      self.node_distances = []
#      self.connected_nodes = []
#      for i, _ in enumerate(self.node_names):
#        self.node_diffs2D.append(self.node_coords - self.node_coords[i])
#        self.connected_nodes.append(
#            np.array([np.where(self.node_names == edge.node)[0][0] for edge in msg.nodes[i].edges]))
#
#      self.node_diffs2D = np.array(self.node_diffs2D)
#      self.node_distances = np.sqrt(np.sum(self.node_diffs2D ** 2, axis=2))
#
#    else:
#      self.node_diffs2D = []
#      self.node_distances = []
#      self.connected_nodes = []
#      self.node_names = []
#      self.node_coords = np.array([])


class TopologicalMap():

  def __init__(self, msg=None):
    # Create from a TopoMap2 msg
    if msg is not None:
      data = json.loads(msg.data)
      self.node_names = np.array([node["node"]["name"] for node in data["nodes"]])
      self.node_coords = np.array([[node["node"]["pose"]["position"]["x"], node["node"]["pose"]["position"]["y"]] for node in data["nodes"]])
      
      edges = [node["node"]["edges"] for node in data["nodes"]]
      self.node_diffs2D = []
      self.node_distances = []
      self.connected_nodes = []
      for i, _ in enumerate(self.node_names):
        self.node_diffs2D.append(self.node_coords - self.node_coords[i])
        self.connected_nodes.append(
            np.array([np.where(self.node_names == edge["node"])[0][0] for edge in edges[i]]))

      self.node_diffs2D = np.array(self.node_diffs2D)
      self.node_distances = np.sqrt(np.sum(self.node_diffs2D ** 2, axis=2))

      self.node_rows = -1 * np.ones(len(self.node_names), dtype=int)
      self.node_tunnels = -1 * np.ones(len(self.node_names), dtype=int)
      for i, node in enumerate(data["nodes"]):
        if "tag" in node["meta"]:
          for tag in node["meta"]["tag"]:
            if "row" in tag:
              self.node_rows[i] = int(tag[-5:])
            elif "tunnel" in tag:
              self.node_tunnels[i] = int(tag[-5:])

    # Create an empty map
    else:
      self.node_diffs2D = np.array([])
      self.node_distances = np.array([])
      self.connected_nodes = np.array([])
      self.node_names = np.array([])
      self.node_coords = np.array([])
      self.node_rows = np.array([])
      self.node_tunnels = np.array([])
