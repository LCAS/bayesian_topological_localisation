#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSProfile
from std_msgs.msg import String
import json
import pyproj

from bayesian_topological_localisation_msgs.srv import LocaliseAgent, UpdatePoseObservation, RestrictMap


class Agent():

  def __init__(self, nh, name):
    self.name = name
    self.cli_register_agent = nh.create_client(srv_type=LocaliseAgent,
                                               srv_name="/bayesian_topological_localisation/localise_agent")
    self.cli_pose_obs = nh.create_client(srv_type=UpdatePoseObservation,
                                         srv_name="/{0}/update_pose_obs".format(self.name))
    self.cli_restr_map = nh.create_client(srv_type=RestrictMap,
                                          srv_name="/{0}/restrict_map".format(self.name))

  def localise(self, prediction_rate=10.0):
    req = LocaliseAgent.Request()
    req.name = self.name
    req.prediction_rate = prediction_rate
    future = self.cli_register_agent.call_async(req)
    return future

  def send_gps(self, x, y):
    req = UpdatePoseObservation.Request()
    req.pose.pose.pose.position.x = x
    req.pose.pose.pose.position.y = y
    req.pose.pose.pose.position.z = 0.2
    req.identifying = True
    future = self.cli_restr_map.call_async(req)
    return future

  def restrict_map(self, row):
    req = RestrictMap.Request()
    req.row = row
    future = self.cli_pose_obs.call_async(req)
    return future


class TPFCARHandler(Node):

  def __init__(self):
    super().__init__("tpf_car_handler")
    self.logger = self.get_logger()

    # Interface with CAR WS
    qos_profile = QoSProfile(depth=1, durability=QoSDurabilityPolicy.VOLATILE)
    self.sub_new_agent = self.create_subscription(String, "/car/new_agent", self.cb_new_agent, qos_profile)
    self.sub_states = self.create_subscription(String, "/car_client/get_states", self.cb_states, qos_profile)
    self.sub_states_kv = self.create_subscription(String, "/car_client/get_states_kv", self.cb_states_kv, qos_profile)
    self.sub_gps = self.create_subscription(String, "/car_client/get_gps", self.cb_gps, qos_profile)

    # Interface with TPF
    self.proj = pyproj.Proj(proj="utm", zone=30, ellps="WGS84", units="m")
    self.map_zero = self.proj(-0.524509505881, 53.268642038)
    self.agents = []

    # Scheduling
    self.client_futures = []

  def spin(self):
    # Work through all waiting futures and remove completed ones
    while rclpy.ok():
      rclpy.spin_once(self)
      incomplete_futures = []
      for f in self.client_futures:
        if not f.done():
          incomplete_futures.append(f)
      self.client_futures = incomplete_futures

  def track_new_agent(self, name):
    # """Register a new CAR agent with TPF"""
    self.logger.info("Registering new agent {0} with TPF...".format(name))
    agent = Agent(self, name)
    self.agents.append(agent)
    future = agent.localise()
    self.client_futures.append(future)

  def cb_new_agent(self, msg):
    # """New agent registered with CAR server"""
    # First check if we already have this agent registered
    new_agent_name = msg.data
    for agent in self.agents:
      if agent.name == new_agent_name:
        return
    # If not, register new agent
    self.track_new_agent(new_agent_name)

  def cb_states(self, msg):
    # """Check if all current CAR agents are being tracked, add missing ones to TPF"""
    data = json.loads(msg.data)
    tracked_agents = [agent.name for agent in self.agents]
    for agent in data["states"]:
      if agent not in tracked_agents:
        self.track_new_agent(agent)

  def cb_states_kv(self, msg):
    self.logger.info("CB states kv")

  def cb_gps(self, msg):
    # """Send a CAR GPS observation to the TPF"""
    data = json.loads(msg.data)
    for agent in self.agents:
      if agent.name == data["user"]:
        lat = data["lat"]
        lon = data["long"]
        trolley_loc = self.proj(lon, lat)
        x = trolley_loc[0] - self.map_zero[0]
        y = trolley_loc[1] - self.map_zero[1]
        self.logger.info("Agent \'{}\' lat/lon: {:.6f}, {:.6f} | x/y: {:.6f}, {:.6f}".format(data["user"], lat, lon, x, y))
        future = agent.send_gps(x, y)
        self.client_futures.append(future)

        if data["row"] != '':
          future = agent.restrict_map(int(data["row"]))
          self.client_futures.append(future)


def main(args=None):
  rclpy.init(args=args)
  tpf_car_handler = TPFCARHandler()
  tpf_car_handler.spin()
  tpf_car_handler.destroy_node()
  rclpy.shutdown()


if __name__ == '__main__':
  main()
