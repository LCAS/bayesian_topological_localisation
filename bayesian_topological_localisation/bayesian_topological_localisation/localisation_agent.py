#!/usr/bin/env python3

# ROS
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSProfile

# Python
import numpy as np
import threading

# Bayesian Topological Localisation
from bayesian_topological_localisation.particle_filter import TopologicalParticleFilter
from bayesian_topological_localisation.prediction_model import PredictionModel
from bayesian_topological_localisation.topological_map import TopologicalMap
from bayesian_topological_localisation_msgs.srv import UpdatePoseObservation, \
                                                       UpdateLikelihoodObservation,  UpdatePriorLikelihoodObservation, \
                                                       Predict, SetFloat64

# Interfaces
from bayesian_topological_localisation_msgs.msg import DistributionStamped, PoseObservation, LikelihoodObservation, ParticlesState
from std_msgs.msg import String
from topological_navigation_msgs.msg import TopologicalMap as TopologicalMapMsg
from visualization_msgs.msg import Marker, MarkerArray


class LocalisationAgent(Node):
  # """A single agent, e.g. a picker or a robot, to be localised"""

  def __init__(self, name="unknown", n_particles=300, do_prediction=True, prediction_rate=0.5, initial_spread_policy=0, prediction_speed_decay=1.0, topo_map=TopologicalMap()):
    super().__init__("bayesian_topological_localisation_agent_{:s}".format(name))
    self.logger = self.get_logger()
    self.logger.info("Creating new localisation agent: {:s}".format(name))

    # Parameters
    self.name = name
    self.n_particles = n_particles
    self.do_prediction = do_prediction
    self.prediction_rate = prediction_rate
    self.initial_spread_policy = initial_spread_policy
    self.prediction_speed_decay = prediction_speed_decay
    self.topo_map = topo_map

    # Publishers
    #self.pub_loc = None
    #self.pub_vis = None

    # Initialise messages
    self.msg_node_marker = Marker()
    self.msg_node_marker = Marker()
    self.msg_node_marker.header.frame_id = "map"
    self.msg_node_marker.type = self.msg_node_marker.SPHERE
    self.msg_node_marker.pose.position.z = 6.0
    self.msg_node_marker.pose.orientation.w = 1.0
    self.msg_node_marker.scale.x = 0.5
    self.msg_node_marker.scale.y = 0.5
    self.msg_node_marker.scale.z = 0.5
    self.msg_node_marker.color.a = 1.0
    self.msg_node_marker.color.r = 0.0
    self.msg_node_marker.color.g = 0.0
    self.msg_node_marker.color.b = 1.0
    self.msg_node_marker.id = 0

    self.msg_particle_marker_array = MarkerArray()
    for i in range(n_particles):    
      msg_particle_marker = Marker()
      msg_particle_marker.header.frame_id = "map"
      msg_particle_marker.type = msg_particle_marker.SPHERE
      msg_particle_marker.pose.position.z = 0.0
      msg_particle_marker.pose.orientation.w = 1.0
      msg_particle_marker.scale.x = 0.1
      msg_particle_marker.scale.y = 0.1
      msg_particle_marker.scale.z = 0.1
      msg_particle_marker.color.a = 0.6
      msg_particle_marker.color.r = 1.0
      msg_particle_marker.color.g = 0.0
      msg_particle_marker.color.b = 0.0
      msg_particle_marker.id = i
      self.msg_particle_marker_array.markers.append(msg_particle_marker)

    self.msg_stateless_particle_marker_array = MarkerArray()
    for i in range(n_particles):
      msg_stateless_particle_marker = Marker()
      msg_stateless_particle_marker.header.frame_id = "map"
      msg_stateless_particle_marker.type = msg_stateless_particle_marker.SPHERE
      msg_stateless_particle_marker.pose.position.z = 0.0
      msg_stateless_particle_marker.pose.orientation.w = 1.0
      msg_stateless_particle_marker.scale.x = 0.1
      msg_stateless_particle_marker.scale.y = 0.1
      msg_stateless_particle_marker.scale.z = 0.1
      msg_stateless_particle_marker.color.a = 0.6
      msg_stateless_particle_marker.color.r = 1.0
      msg_stateless_particle_marker.color.g = 1.0
      msg_stateless_particle_marker.color.b = 0.0
      msg_stateless_particle_marker.id = i
      self.msg_stateless_particle_marker_array.markers.append(msg_stateless_particle_marker)

    # Initialize the prediction model
    self.pm = PredictionModel(pred_type=PredictionModel.CTMC,
                              node_coords=self.topo_map.node_coords,
                              node_diffs2D=self.topo_map.node_diffs2D,
                              node_distances=self.topo_map.node_distances,
                              connected_nodes=self.topo_map.connected_nodes)

    # Default values for pf
    default_reinit_jsd_threshold = 0.975
    default_unconnected_jump_threshold = 0.6

    # Initialize a new instance of particle_filter
    self.tpf = TopologicalParticleFilter(num=n_particles,
                                         prediction_model=self.pm,
                                         initial_spread_policy=initial_spread_policy,
                                         prediction_speed_decay=prediction_speed_decay,
                                         node_coords=self.topo_map.node_coords,
                                         node_distances=self.topo_map.node_distances,
                                         connected_nodes=self.topo_map.connected_nodes,
                                         node_diffs2D=self.topo_map.node_diffs2D,
                                         node_names=self.topo_map.node_names,
                                         reinit_jsd_threshold=default_reinit_jsd_threshold,
                                         unconnected_jump_threshold=default_unconnected_jump_threshold)

    # Publishers
    self.pub_cn = self.create_publisher(String, "~/estimated_node", 10)
    self.pub_pd = self.create_publisher(DistributionStamped, "~/current_prob_dist", 10)
    self.pub_ptcs = self.create_publisher(ParticlesState, "~/particles_states", 10)
    self.pub_cnviz = self.create_publisher(Marker, "~/estimated_node_viz", 10)
    self.pub_parviz = self.create_publisher(MarkerArray, "~/particles_viz", 10)
    self.pub_staparviz = self.create_publisher(MarkerArray, "~/stateless_particles_viz", 10)

    # Subscribers
    qos_profile = QoSProfile(depth=1, durability=QoSDurabilityPolicy.VOLATILE)
    self.sub_pose_observation = self.create_subscription(PoseObservation, "~/pose_obs", self.cb_pose_obs, qos_profile)
    self.sub_likelihood_observation = self.create_subscription(LikelihoodObservation, "~/likelihood_obs", self.cb_likelihood_obs, qos_profile)

    # Services
    self.srv_update_pose_observation = self.create_service(UpdatePoseObservation, "~/update_pose_obs", self.handler_update_pose),
    self.srv_update_likelihood_observation = self.create_service(UpdateLikelihoodObservation, "~/update_likelihood_obs", self.handler_update_likelihood),
    self.srv_predict_stateless = self.create_service(Predict, "~/predict_stateless", self.handler_do_stateless_prediction),
    self.srv_update_stateless = self.create_service(UpdatePriorLikelihoodObservation, "~/update_stateless", self.handler_do_stateless_update)

    # Timers
    self.tmr_predict = self.create_timer(1.0 / prediction_rate, self.cb_predict)

  def cb_predict(self):
    if self.do_prediction:
      self.logger.info("Doing prediction for agent {:s}".format(self.name))
      p_estimate, particles = self.tpf.predict(timestamp_secs=self.get_clock().now().nanoseconds / 1e9)
      if not (p_estimate is None or particles is None): 
        self.publish(p_estimate.node, particles)

  def prepare_pd_msg(self, particles, timestamp=None):
    msg_pd = DistributionStamped()
    nodes = [p.node for p in particles]
    nodes, counts = np.unique(nodes, return_counts=True)

    probs = np.zeros((self.topo_map.node_names.shape[0]))
    probs[nodes] = counts.astype(float) / np.sum(counts)

    if timestamp is None:
      timestamp = self.get_clock().now()

    msg_pd.header.stamp = timestamp.to_msg()
    msg_pd.nodes = self.topo_map.node_names.tolist()
    msg_pd.values = np.copy(probs).tolist()

    return msg_pd

  def prepare_cn_msg(self, node):
    msg_str = String()
    msg_str.data = self.topo_map.node_names[node]
    return msg_str

  def prepare_ptcs_msg(self, particles):
    msg_particles = ParticlesState()
    msg_particles.nodes = [self.topo_map.node_names[p.node] for p in particles]
    msg_particles.vels_x = [p.vel[0] for p in particles]
    msg_particles.vels_y = [p.vel[1] for p in particles]
    msg_particles.times = [p.life for p in particles]
    return msg_particles

  # function to publish current node and particles distribution
  def publish(self, node, particles):
    self.pub_cn.publish(self.prepare_cn_msg(node))
    self.pub_pd.publish(self.prepare_pd_msg(particles))
    self.pub_ptcs.publish(self.prepare_ptcs_msg(particles))

    # publish viz stuff
    for i, p in enumerate(particles):
      self.msg_particle_marker_array.markers[i].header.stamp = self.get_clock().now().to_msg()
      self.msg_particle_marker_array.markers[i].pose.position.x = self.topo_map.node_coords[p.node][0] + \
        self.msg_particle_marker_array.markers[i].scale.x * np.random.randn(1, 1)
      self.msg_particle_marker_array.markers[i].pose.position.y = self.topo_map.node_coords[p.node][1] + \
        self.msg_particle_marker_array.markers[i].scale.y * np.random.randn(1, 1)
    self.msg_node_marker.pose.position.x = self.topo_map.node_coords[node][0]
    self.msg_node_marker.pose.position.y = self.topo_map.node_coords[node][1]

    self.pub_cnviz.publish(self.msg_node_marker)
    self.pub_parviz.publish(self.msg_particle_marker_array)

  # function to publish stateless particles distribution viz
  def publish_stateless_viz(self, particles):
    # publish viz stuff
    for i, p in enumerate(particles):
      self.msg_stateless_particle_marker_array.markers[i].header.stamp = self.get_clock().now().to_msg()
      self.msg_stateless_particle_marker_array.markers[i].pose.position.x = self.topo_map.node_coords[p.node][0] + \
        self.msg_stateless_particle_marker_array.markers[i].scale.x * np.random.randn(1, 1)
      self.msg_stateless_particle_marker_array.markers[i].pose.position.y = self.topo_map.node_coords[p.node][1] + \
        self.msg_stateless_particle_marker_array.markers[i].scale.y * np.random.randn(1, 1)

    self.pub_staparviz.publish(self.msg_stateless_particle_marker_array)

  ## topic callbacks ##
  # send pose observation to particle filter
  def cb_pose_obs(self, msg):
    if np.isfinite(msg.pose.pose.pose.position.x) and \
       np.isfinite(msg.pose.pose.pose.position.y) and \
       np.isfinite(msg.pose.pose.covariance[0]) and \
       np.isfinite(msg.pose.pose.covariance[7]):

      p_estimated, particles = self.tpf.receive_pose_obs(msg.pose.pose.pose.position.x,
                                                         msg.pose.pose.pose.position.y,
                                                         msg.pose.pose.covariance[0],  # x variance
                                                         msg.pose.pose.covariance[7],  # y variance
                                                         self.get_clock().now().nanoseconds / 1e9,
                                                         identifying=msg.identifying)
      self.publish(p_estimated.node, particles)

    else:
      self.logger.warn(
          "Received non-admissible pose observation <{}, {}, {}, {}>, discarded".format(msg.pose.pose.pose.position.x, msg.pose.pose.pose.position.y, msg.pose.pose.covariance[0], msg.pose.pose.covariance[7]))

  # send likelihood observation to particle filter
  def cb_likelihood_obs(self, msg):
    if len(msg.likelihood.nodes) == len(msg.likelihood.values):
      try:
        nodes = [np.where(self.topo_map.node_names == nname)[0][0] for nname in msg.likelihood.nodes]
      except IndexError:
        self.logger.warn(
            "Received non-admissible node name {}, likelihood discarded".format(msg.likelihood.nodes))
      else:
        values = np.array(msg.likelihood.values)
        if np.isfinite(values).all() and (values >= 0.).all() and np.sum(values) > 0:
          p_estimated, particles = self.tpf.receive_likelihood_obs(nodes, 
                                                                   msg.likelihood.values,
                                                                   self.get_clock().now().nanoseconds / 1e9,
                                                                   identifying=msg.identifying)
          self.publish(p_estimated.node, particles)
        else:
          self.logger.warn(
              "Received non-admissible likelihood observation {}, discarded".format(msg.likelihood.values))
    else:
      self.logger.warn(
          "Nodes array and values array sizes do not match {} != {}, discarding likelihood observation".format(len(msg.likelihood.nodes), len(msg.likelihood.values)))

  ## Services handlers ##
  # Get the pose observation and returns the localisation result
  def handler_update_pose(self, request, response):
    if np.isfinite(request.pose.pose.pose.position.x) and \
       np.isfinite(request.pose.pose.pose.position.y) and \
       np.isfinite(request.pose.pose.covariance[0]) and \
       np.isfinite(request.pose.pose.covariance[7]):

      p_estimated, particles = self.tpf.receive_pose_obs(request.pose.pose.pose.position.x,
                                                         request.pose.pose.pose.position.y,
                                                         request.pose.pose.covariance[0],  # x variance
                                                         request.pose.pose.covariance[7],  # y variance
                                                         self.get_clock().now().nanoseconds / 1e9,
                                                         identifying=request.identifying)
      self.publish(p_estimated.node, particles)
      response.success = True
      response.estimated_node = self.prepare_cn_msg(p_estimated.node).data
      response.current_prob_dist = self.prepare_pd_msg(particles)
      return(response)
    else:
      self.logger.warn(
          "Received non-admissible pose observation <{}, {}, {}, {}>, discarded".format(request.pose.pose.pose.position.x, request.pose.pose.pose.position.y, request.pose.pose.covariance[0], request.pose.pose.covariance[7]))
    
    # fallback negative response
    response.success = False
    return(response)

  # get a likelihood observation and return localisation result
  def handler_update_likelihood(self, request, response):
    if len(request.likelihood.nodes) == len(request.likelihood.values):
      try:
        nodes = [np.where(self.topo_map.node_names == nname)[0][0] for nname in request.likelihood.nodes]
      except IndexError:
        self.logger.warn("Received non-admissible node name {}, likelihood discarded".format(request.likelihood.nodes))
      else:
        values = np.array(request.likelihood.values)
        # self.logger.info("Received likelihood: {}".format(zip(nodes, values)))
        if np.isfinite(values).all() and (values >= 0.).all() and np.sum(values) > 0:
          p_estimated, particles = self.tpf.receive_likelihood_obs(nodes,
                                                                   request.likelihood.values,
                                                                   self.get_clock().now().nanoseconds / 1e9,
                                                                   identifying=request.identifying)
          self.publish(p_estimated.node, particles)
          response.success = True
          response.estimated_node = self.prepare_cn_msg(p_estimated.node).data
          response.current_prob_dist = self.prepare_pd_msg(particles)
          return(response)
        else:
          self.logger.warn("Received non-admissible likelihood observation {}, discarded".format(request.likelihood.values))

    else:
      self.logger.warn("Nodes array and values array sizes do not match {} != {}, discarding likelihood observation".format(
          len(request.likelihood.nodes), len(request.likelihood.values)))

    # fallback negative response
    response.success = False
    return(response)

  def handler_do_stateless_prediction(self, request, response):
    # """Predict based on current particle filter without modifying its state"""
    _tpf = self.tpf.copy()

    # if requested pred rate is lesseq than 0 use the global one
    # TODO: Check all service parameters by default somewhere
    #_prediction_rate = (request.prediction_rate, prediction_rate)[
    #  request.prediction_rate <= 0.]
    pred_step_secs = 1. / request.prediction_rate
    secs_left = max(0.0, request.secs_from_now)
    time = self.get_clock().now()
    response.success = True

    # sub-function to append predictions to the result message
    def ___append_prediction(node, particles, secs_passed, timestamp):
      if not (node is None or particles is None):
        self.publish_stateless_viz(particles)
        response.secs_from_now.append(secs_passed)
        response.estimated_node.append(self.prepare_cn_msg(node).data)
        response.prob_dist.append(self.prepare_pd_msg(particles, timestamp=timestamp))
        return True
      else:
        response.success = False
        self.logger.warn(
          "Cannot perform prediction, no observation received so far.")
        return False

    # perform all the steps until reached limit time
    while secs_left > 0:
      p_estimated, particles = _tpf.predict(
        timestamp_secs=time.to_sec()
      )
      if request.return_history:
        succ = ___append_prediction(
            p_estimated.node, 
            particles,
            secs_passed=request.secs_from_now - secs_left,
            timestamp=time)
        if not succ:
          break
        
      time += rclpy.time.Duration(seconds=pred_step_secs)
      secs_left -= pred_step_secs

    # add last prediction at secs_left == 0
    p_estimated, particles = _tpf.predict(
      timestamp_secs=time.to_sec()
    )
    _ = ___append_prediction(
          p_estimated.node, 
          particles,
          secs_passed=request.secs_from_now - secs_left,
          timestamp=time)

    return response

  def handler_do_stateless_update(self, request, response):
    # create a new PF and assign the prior distribution to start up with
    _tpf = TopologicalParticleFilter(num=self.n_particles,
                                     prediction_model=self.pm,
                                     initial_spread_policy=self.initial_spread_policy,
                                     prediction_speed_decay=self.prediction_speed_decay,
                                     node_coords=self.topo_map.node_coords,
                                     node_distances=self.topo_map.node_distances,
                                     connected_nodes=self.topo_map.connected_nodes,
                                     node_diffs2D=self.topo_map.node_diffs2D,
                                     node_names=self.topo_map.node_names)
    _tpf.print_debug = False
    if len(request.likelihood.nodes) == len(request.likelihood.values) and \
        len(request.prior.nodes) == len(request.prior.values):
      try:
        lkl_nodes = [np.where(self.topo_map.node_names == nname)[0][0] for nname in request.likelihood.nodes]
        pr_nodes = [np.where(self.topo_map.node_names == nname)[0][0] for nname in request.prior.nodes]
      except IndexError:
        self.logger.warn(
          "Received non-admissible node name {}/{}, prior/likelihood discarded".format(request.prior.nodes, request.likelihood.nodes))
      else:
        pr_values = np.array(request.prior.values)
        # self.logger.info(
        #     "Received prior: {}".format(zip(pr_nodes, pr_values)))
        lkl_values = np.array(request.likelihood.values)
        # self.logger.info(
        #     "Received likelihood: {}".format(zip(lkl_nodes, lkl_values)))
        if np.isfinite(lkl_values).all() and (lkl_values >= 0.).all() and np.sum(lkl_values) > 0 and \
            np.isfinite(pr_values).all() and (pr_values >= 0.).all() and np.sum(pr_values) > 0:
          ts = self.get_clock().now().nanoseconds / 1e9
          # send the prior (assignment bcs it's the first observation ever received)
          _, _ = _tpf.receive_likelihood_obs(
            pr_nodes,
            request.prior.values,
            ts,
            True # this has to be true for initializing the distribution
          )
          # send the lkl (prediction is not performed here because the ts is the same as prior ts)
          p_estimated, particles = _tpf.receive_likelihood_obs(
            lkl_nodes,
            request.likelihood.values,
            ts,
            False  # this has to be false to avoid the risk it gets re-initialised uniformly by the JSD threshold
          )
          response.success = True
          response.estimated_node = self.prepare_cn_msg(p_estimated.node).data
          response.current_prob_dist = self.prepare_pd_msg(particles)
          return(response)
        else:
          self.logger.warn(
              "Received non-admissible prior/likelihood observation {}, discarded".format(request.prior.values, request.likelihood.values))
    else:
      self.logger.warn(
          "Nodes array and values array sizes do not match {} != {}/{} != {}, discarding prior/likelihood observation".format(len(request.prior.nodes), len(request.prior.values), len(request.likelihood.nodes), len(request.likelihood.values)))

