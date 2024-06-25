#!/usr/bin/env python3

import numpy as np
import json


class TopologicalMap():

  def __init__(self, msg=None):
    # Row / Tunnel number we are restricted to
    self.row_restr = -1
    self.tunnel_restr = -1

    # Create from a TopoMap2 msg
    if msg is not None:
      data = json.loads(msg.data)
      self.node_names_ = np.array([node["node"]["name"] for node in data["nodes"]])
      self.node_coords_ = np.array([[node["node"]["pose"]["position"]["x"], node["node"]["pose"]["position"]["y"]] for node in data["nodes"]])
      
      edges = [node["node"]["edges"] for node in data["nodes"]]
      self.node_diffs2D_ = []
      self.node_distances_ = []
      self.connected_nodes_ = []
      for i, _ in enumerate(self.node_names_):
        self.node_diffs2D_.append(self.node_coords_ - self.node_coords_[i])
        self.connected_nodes_.append(
            np.array([np.where(self.node_names_ == edge["node"])[0][0] for edge in edges[i]]))

      self.node_diffs2D_ = np.array(self.node_diffs2D_)
      self.node_distances_ = np.sqrt(np.sum(self.node_diffs2D_ ** 2, axis=2))

      self.node_rows_ = -1 * np.ones(len(self.node_names_), dtype=int)
      self.node_tunnels_ = -1 * np.ones(len(self.node_names_), dtype=int)
      for i, node in enumerate(data["nodes"]):
        if "tag" in node["meta"]:
          for tag in node["meta"]["tag"]:
            if "row" in tag:
              self.node_rows_[i] = int(tag[-5:])
            elif "tunnel" in tag:
              self.node_tunnels_[i] = int(tag[-5:])

    # Create an empty map
    else:
      self.node_names_ = np.array([])
      self.node_coords_ = np.array([])
      self.node_diffs2D_ = np.array([])
      self.node_distances_ = np.array([])
      self.connected_nodes_ = np.array([])
      self.node_rows_ = np.array([])
      self.node_tunnels_ = np.array([])

  def node_names(self, ignore_restrictions=False):
    if ignore_restrictions or (self.row_restr == -1 and self.tunnel_restr == -1):
      return self.node_names_
    else:
      return self.node_names_restr_

  def node_coords(self, ignore_restrictions=False):
    if ignore_restrictions or (self.row_restr == -1 and self.tunnel_restr == -1):
      return self.node_coords_
    else:
      return self.node_coords_restr_

  def node_diffs2D(self, ignore_restrictions=False):
    if ignore_restrictions or (self.row_restr == -1 and self.tunnel_restr == -1):
      return self.node_diffs2D_
    else:
      return self.node_diffs2D_restr_

  def node_distances(self, ignore_restrictions=False):
    if ignore_restrictions or (self.row_restr == -1 and self.tunnel_restr == -1):
      return self.node_distances_
    else:
      return self.node_distances_restr_

  def connected_nodes(self, ignore_restrictions=False):
    if ignore_restrictions or (self.row_restr == -1 and self.tunnel_restr == -1):
      return self.connected_nodes_
    else:
      return self.connected_nodes_restr_

  def node_rows(self, ignore_restrictions=False):
    if ignore_restrictions or (self.row_restr == -1 and self.tunnel_restr == -1):
      return self.node_rows_
    else:
      return self.node_rows_restr_

  def restrict(self, row=-1, tunnel=-1):
    # Restrict by row
    idcs_row = np.arange(len(self.node_names_))
    if row != -1:
      self.row_restr = row
      idcs_row = np.where(self.node_rows_ == row)[0]

    # Restrict by tunnel
    idcs_tunnel = np.arange(len(self.node_names_))
    if tunnel != -1:
      self.tunnel_restr = tunnel
      idcs_tunnel = np.where(self.node_tunnels_ == tunnel)[0]
    
    idcs_restr = np.intersect1d(idcs_row, idcs_tunnel)

    # Only restrict if any nodes are left
    if len(idcs_restr) == 0:
      return False

    self.node_names_restr_ = self.node_names_[idcs_restr]
    self.node_coords_restr_ = self.node_coords_[idcs_restr]
    self.node_diffs2D_restr_ = self.node_diffs2D_[np.ix_(idcs_restr, idcs_restr)]
    self.node_distances_restr_ = self.node_distances_[np.ix_(idcs_restr, idcs_restr)]
    self.connected_nodes_restr_ = [cn for idx, cn in enumerate(self.connected_nodes_) if idx in idcs_restr]
    self.connected_nodes_restr_ = [np.array([np.where(idcs_restr == cn)[0][0] for cn in cns if cn in idcs_restr]) for cns in self.connected_nodes_restr_]
    self.node_tunnels_restr_ = self.node_tunnels_[idcs_restr]
    self.node_rows_restr_ = self.node_rows_[idcs_restr]
      
    return True