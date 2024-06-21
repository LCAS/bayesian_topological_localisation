#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSProfile
import time
import numpy as np

from bayesian_topological_localisation.particle_filter import TopologicalParticleFilter
from bayesian_topological_localisation.prediction_model import PredictionModel
from bayesian_topological_localisation.localisation_agent import LocalisationAgent
from bayesian_topological_localisation.topological_map import TopologicalMap
from bayesian_topological_localisation_msgs.srv import LocaliseAgent, StopLocalise, UpdatePoseObservation, \
                                                       UpdateLikelihoodObservation,  UpdatePriorLikelihoodObservation, \
                                                       Predict, SetFloat64
from bayesian_topological_localisation_msgs.msg import DistributionStamped, PoseObservation, LikelihoodObservation, ParticlesState
from visualization_msgs.msg import Marker, MarkerArray
from topological_navigation_msgs.msg import TopologicalMap as TopologicalMapMsg
from std_msgs.msg import String


class TopologicalLocalisation(Node):
  # """."""

  def __init__(self):
    super().__init__("bayesian_topological_localisation")
    self.logger = self.get_logger()

    # Agents currently being tracked
    self.agents = []

    # Info about topology
    self.topo_map = None

    # Subscribe with transient QoS so previously published topomap gets loaded
    qos_profile = QoSProfile(depth=1, durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)
    self.sub_topo_map = self.create_subscription(String, "/topological_map_2", self.cb_topo_map, qos_profile)

    self.logger.info("Waiting for topological map...")
    while self.topo_map is None:
      rclpy.spin_once(self)
      time.sleep(0.5)

    # Declare services
    self.srv_localise_agent = self.create_service(LocaliseAgent, "~/localise_agent", self.handler_localise_agent)
    self.srv_stop_localise = self.create_service(StopLocalise, "~/stop_localise", self.handler_stop_localise)
    self.srv_set_jsd_upper = self.create_service(SetFloat64, "~/set_JSD_upper_bound", self.handler_set_JSD_upper_bound)
    self.srv_set_entropy_lower = self.create_service(SetFloat64, "~/set_entropy_lower_bound", self.handler_set_entropy_lower_bound)

    # Executor taking care of multiple agents
    self.thread_executor = rclpy.executors.MultiThreadedExecutor()
    self.thread_executor.add_node(self)
    self.logger.info("DONE")

  def run(self):
    # """ Create an executor spinning this node and all agents """
    self.thread_executor.spin()

  def handler_set_JSD_upper_bound(self, request, response):
    for a in self.agents:
      a.tpf.set_JSD_upper_bound(request.value)

    response.success = True
    return response

  def handler_set_entropy_lower_bound(self, request, response):
    for a in self.agents:
      a.tpf.set_entropy_lower_bound(request.value)

    response.success = True
    return response

  def handler_localise_agent(self, request, response):
    # """Register new agent to localise"""
    self.logger.info("Received request to localise new agent {}".format(request.name))

    # Do not localise same agent twice
    if request.name in [a.name for a in self.agents]:
      self.logger.warn("Agent {} already being localised, not registering agent twice.".format(request.name))
      response.success = False
      return response

    # Create a localisation agent
    agent = LocalisationAgent(name=request.name, 
                              n_particles=request.n_particles, 
                              do_prediction=request.do_prediction, 
                              prediction_rate=request.prediction_rate, 
                              topo_map=self.topo_map)

    # Add an agent to the threading execution
    self.agents.append(agent)
    self.thread_executor.add_node(agent)

    self.logger.info("DONE")
    response.success = True
    return response

  def handler_stop_localise(self, request, response):
    # """Remove specific localisation agent."""
    self.logger.info("Unregistering agent {} for localisation".format(request.name))

    # default name is unknown if requested is ''
    name = (request.name, 'unknown')[request.name == '']
    if name in [a.name for a in self.agents]:

      # Remove agent node from executed threads, kill the ROS node and delete reference 
      agent_idx = [a.name for a in self.agents].index(name)
      self.thread_executor.remove_node(self.agents[agent_idx])
      self.agents[agent_idx].destroy_node()
      del self.agents[agent_idx]

      self.logger.info("DONE")
      response.success = True
      return response

    else:
      self.logger.warn("No agent called {} is is being localised.".format(name))
      response.success = False
      return response

  def cb_topo_map(self, msg):
    """Receive the Topological Map."""
    self.topo_map = TopologicalMap(msg)
    self.logger.info("Received topomap2")


def main(args=None):
  rclpy.init(args=args)
  localisation_node = TopologicalLocalisation()
  localisation_node.run()
  rclpy.shutdown()


if __name__ == "__main__":
  main()
