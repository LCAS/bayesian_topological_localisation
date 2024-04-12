#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSProfile
import threading
import time
import numpy as np

from bayesian_topological_localisation.particle_filter import TopologicalParticleFilter
from bayesian_topological_localisation.prediction_model import PredictionModel
from bayesian_topological_localisation_msgs.srv import LocaliseAgent, StopLocalise, UpdatePoseObservation, \
                                                       UpdateLikelihoodObservation,  UpdatePriorLikelihoodObservation, \
                                                       Predict, SetFloat64
from bayesian_topological_localisation_msgs.msg import DistributionStamped, PoseObservation, LikelihoodObservation, ParticlesState
from visualization_msgs.msg import Marker, MarkerArray
from topological_navigation_msgs.msg import TopologicalMap
from std_msgs.msg import String


class TopologicalLocalisation(Node):

  def __init__(self):
    super().__init__("bayesian_topological_localisation")
    self.logger = self.get_logger()

    # agents currently tracking
    self.agents = []
    # observation subscribers for each agent
    self.obs_subscribers = []
    # # publishers localisation result for each agent
    self.res_publishers = []
    # # publishers viz result for each agent
    self.viz_publishers = []
    # services for updating the state estimation
    self.upd_services = []
    # thread that loop predictions at fixed rate for each agent
    self.prediction_threads = []
    # contains the particle filters for each agent
    self.pfs = []

    # these will contain info about the topology
    self.topo_map = None
    self.node_diffs2D = []
    self.node_distances = []
    self.connected_nodes = []
    self.node_names = []
    self.node_coords = np.array([])

    # contains a list of threading.Event for stopping the localisation of each agent
    self.stopping_events = []

    # to avoid inconsistencies when registering/unregistering agents concurrently
    self.internal_lock = threading.Lock()

    # default values for pf
    self.default_reinit_jsd_threshold = 0.975
    self.default_unconnected_jump_threshold = 0.6

    # Subscribe with transient QoS so old topomap gets loaded
    qos_profile = QoSProfile(depth=1, durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)
    self.sub_topo_map = self.create_subscription(TopologicalMap, "/topological_map", self._topo_map_cb, qos_profile)

    self.logger.info("Waiting for topological map...")
    while self.topo_map is None:
      rclpy.spin_once(self)
      time.sleep(0.5)
      
    # declare services
    self.srv_localise_agent = self.create_service(LocaliseAgent, "~/localise_agent", self._localise_agent_handler)
    self.srv_stop_localise = self.create_service(StopLocalise, "~/stop_localise", self._stop_localise_handler)
    self.srv_set_jsd_upper = self.create_service(SetFloat64, "~/set_JSD_upper_bound", self._set_JSD_upper_bound)
    self.srv_set_entropy_lower = self.create_service(SetFloat64, "~/set_entropy_lower_bound", self._set_entropy_lower_bound)

    self.logger.info("DONE")

  def _set_JSD_upper_bound(self, request, response):
    self.default_reinit_jsd_threshold = request.value
    for pf in self.pfs:
      pf.set_JSD_upper_bound(request.value)

    response.success = True
    return response

  def _set_entropy_lower_bound(self, request, response):
    self.default_unconnected_jump_threshold = request.value
    for pf in self.pfs:
      pf.set_entropy_lower_bound(request.value)

    response.success = True
    return response

  def _localise_agent_handler(self, request, response):
    # """Register new agent to localise"""
    self.logger.info("Received request to localise new agent {}".format(request.name))
    # lock resources
    self.internal_lock.acquire()

    # to stop executing in other threads/cbs
    stop_event = threading.Event()

    ## set default values ##
    # default name is unknown if requested is ''
    name = (request.name, 'unknown')[request.name == '']
    # default particles number is 300 if requested is 0
    n_particles = (request.n_particles, 300)[request.n_particles <= 0]
    initial_spread_policy = 0
    prediction_model = 0
    do_prediction = request.do_prediction
    # default prediction rate is 0.5 if requested is 0.
    prediction_rate = (request.prediction_rate, 0.5)[
      request.prediction_rate <= 0.]
    # default speed decay is 1 if requested is 0
    prediction_speed_decay = 1.0

    if name in self.agents:
      self.logger.warn("Agent {} already being localised".format(name))
      # release resources
      self.internal_lock.release()
      response.success = False
      return response

    # Initialize the prediction model
    # if prediction_model == LocaliseAgentRequest.PRED_CTMC:
    pm = PredictionModel(
      pred_type=PredictionModel.CTMC,
      node_coords=self.node_coords,
      node_diffs2D=self.node_diffs2D,
      node_distances=self.node_distances,
      connected_nodes=self.connected_nodes
    )
    # elif prediction_model == LocaliseAgentRequest.PRED_IDENTITY:
    #     pm = PredictionModel(
    #         pred_type=PredictionModel.IDENTITY
    #     )
    # else:
    #     rospy.logerr(
    #         "Prediction model {} unknown".format(prediction_model))
    #     # release resources
    #     self.internal_lock.release()
    #     return LocaliseAgentResponse(False)


    # Initialize publishers and messages
    cn_pub = self.create_publisher(String, "{}/estimated_node".format(name), 10)
    pd_pub = self.create_publisher(DistributionStamped, "{}/current_prob_dist".format(name), 10)
    ptcs_pub = self.create_publisher(ParticlesState, "{}/particles_states".format(name), 10)
    self.res_publishers.append((cn_pub, pd_pub, ptcs_pub))
    cnviz_pub = self.create_publisher(Marker, "{}/estimated_node_viz".format(name), 10)
    parviz_pub = self.create_publisher(MarkerArray, "{}/particles_viz".format(name), 10)
    staparviz_pub = self.create_publisher(MarkerArray, "{}/stateless_particles_viz".format(name), 10)
    self.viz_publishers.append((cnviz_pub, parviz_pub, staparviz_pub))
    nodemkrmsg = Marker()
    nodemkrmsg.header.frame_id = "map"
    nodemkrmsg.type = nodemkrmsg.SPHERE
    nodemkrmsg.pose.position.z = 6.0
    nodemkrmsg.pose.orientation.w = 1.0
    nodemkrmsg.scale.x = 0.5
    nodemkrmsg.scale.y = 0.5
    nodemkrmsg.scale.z = 0.5
    nodemkrmsg.color.a = 1.0
    nodemkrmsg.color.r = 0.0
    nodemkrmsg.color.g = 0.0
    nodemkrmsg.color.b = 1.0
    nodemkrmsg.id = 0
    ptcsarrmsg = MarkerArray()
    for i in range(n_particles):
      ptcmkrmsg = Marker()
      ptcmkrmsg.header.frame_id = "map"
      ptcmkrmsg.type = ptcmkrmsg.SPHERE
      ptcmkrmsg.pose.position.z = 0.0
      ptcmkrmsg.pose.orientation.w = 1.0
      ptcmkrmsg.scale.x = 0.1
      ptcmkrmsg.scale.y = 0.1
      ptcmkrmsg.scale.z = 0.1
      ptcmkrmsg.color.a = 0.6
      ptcmkrmsg.color.r = 1.0
      ptcmkrmsg.color.g = 0.0
      ptcmkrmsg.color.b = 0.0
      ptcmkrmsg.id = i
      ptcsarrmsg.markers.append(ptcmkrmsg)
    staptcsarrmsg = MarkerArray()
    for i in range(n_particles):
      staptcmkrmsg = Marker()
      staptcmkrmsg.header.frame_id = "map"
      staptcmkrmsg.type = staptcmkrmsg.SPHERE
      staptcmkrmsg.pose.position.z = 0.0
      staptcmkrmsg.pose.orientation.w = 1.0
      staptcmkrmsg.scale.x = 0.1
      staptcmkrmsg.scale.y = 0.1
      staptcmkrmsg.scale.z = 0.1
      staptcmkrmsg.color.a = 0.6
      staptcmkrmsg.color.r = 1.0
      staptcmkrmsg.color.g = 1.0
      staptcmkrmsg.color.b = 0.0
      staptcmkrmsg.id = i
      staptcsarrmsg.markers.append(staptcmkrmsg)

    # get the how to spread the particles initially
    # if request.initial_spread_policy == LocaliseAgentRequest.CLOSEST_NODE:
    #     sigma = -1
    # elif request.initial_spread_policy == LocaliseAgentRequest.SPREAD_RADIUS:
    #     sigma = request.initial_spread_radius
    # elif request.initial_spread_policy == LocaliseAgentRequest.SPREAD_UNIFORM:
    #     sigma = np.inf
    # else:
    #     sigma = -1

    # Initialize a new instance of particle_filter
    pf = TopologicalParticleFilter(
      num=n_particles,
      prediction_model=pm,
      initial_spread_policy=initial_spread_policy,
      prediction_speed_decay=prediction_speed_decay,
      node_coords=self.node_coords,
      node_distances=self.node_distances,
      connected_nodes=self.connected_nodes,
      node_diffs2D=self.node_diffs2D,
      node_names=self.node_names,
      reinit_jsd_threshold=self.default_reinit_jsd_threshold,
      unconnected_jump_threshold=self.default_unconnected_jump_threshold
    )

    def __prepare_pd_msg(particles, timestamp=None):
      pdmsg = DistributionStamped()
      _nodes = [p.node for p in particles]
      nodes, counts = np.unique(_nodes, return_counts=True)

      probs = np.zeros((self.node_names.shape[0]))
      probs[nodes] = counts.astype(float) / np.sum(counts)

      if timestamp is None:
        timestamp = rclpy.time.get_clock.now()
      pdmsg.header.stamp = timestamp
      pdmsg.nodes = self.node_names.tolist()
      pdmsg.values = np.copy(probs).tolist()

      return pdmsg

    def __prepare_cn_msg(node):
      strmsg = String()
      strmsg.data = self.node_names[node]
      return strmsg

    def __prepare_ptcs_msg(particles):
      ptcsmsg = ParticlesState()
      ptcsmsg.nodes = [self.node_names[p.node] for p in particles]
      ptcsmsg.vels_x = [p.vel[0] for p in particles]
      ptcsmsg.vels_y = [p.vel[1] for p in particles]
      ptcsmsg.times = [p.life for p in particles]

      return ptcsmsg

    # function to publish current node and particles distribution
    def __publish(node, particles):
      if not stop_event.is_set():
        cn_pub.publish(__prepare_cn_msg(node))
        pd_pub.publish(__prepare_pd_msg(particles))
        ptcs_pub.publish(__prepare_ptcs_msg(particles))

        # publish viz stuff
        for i, p in enumerate(particles):
          ptcsarrmsg.markers[i].header.stamp = rclpy.time.get_clock.now()
          ptcsarrmsg.markers[i].pose.position.x = self.node_coords[p.node][0] + \
            ptcsarrmsg.markers[i].scale.x * np.random.randn(1, 1)
          ptcsarrmsg.markers[i].pose.position.y = self.node_coords[p.node][1] + \
            ptcsarrmsg.markers[i].scale.y * np.random.randn(1, 1)
        nodemkrmsg.pose.position.x = self.node_coords[node][0]
        nodemkrmsg.pose.position.y = self.node_coords[node][1]

        cnviz_pub.publish(nodemkrmsg)
        parviz_pub.publish(ptcsarrmsg)

    # function to publish stateless particles distribution viz
    def __publish_stateless_viz(particles):
      if not stop_event.is_set():
        # publish viz stuff
        for i, p in enumerate(particles):
          staptcsarrmsg.markers[i].header.stamp = rclpy.time.get_clock.now()
          staptcsarrmsg.markers[i].pose.position.x = self.node_coords[p.node][0] + \
            staptcsarrmsg.markers[i].scale.x * np.random.randn(1, 1)
          staptcsarrmsg.markers[i].pose.position.y = self.node_coords[p.node][1] + \
            staptcsarrmsg.markers[i].scale.y * np.random.randn(1, 1)

        staparviz_pub.publish(staptcsarrmsg)

    ## topic callbacks ##
    # send pose observation to particle filter
    def __pose_obs_cb(msg):
      if np.isfinite(msg.pose.pose.pose.position.x) and \
          np.isfinite(msg.pose.pose.pose.position.y) and \
          np.isfinite(msg.pose.pose.covariance[0]) and \
          np.isfinite(msg.pose.pose.covariance[7]):
        p_estimated, particles = pf.receive_pose_obs(
          msg.pose.pose.pose.position.x,
          msg.pose.pose.pose.position.y,
          msg.pose.pose.covariance[0], # variance of x
          msg.pose.pose.covariance[7], # variance of y
          # (rclpy.time.get_clock.now().to_sec(), msg.pose.header.stamp.to_sec())[
            # msg.pose.header.stamp.to_sec() > 0]
          rclpy.time.get_clock.now().to_sec(),
          identifying=msg.identifying
        )
        __publish(p_estimated.node, particles)
      else:
        self.logger.warn(
          "Received non-admissible pose observation <{}, {}, {}, {}>, discarded".format(msg.pose.pose.pose.position.x, msg.pose.pose.pose.position.y, msg.pose.pose.covariance[0], msg.pose.pose.covariance[7]))

    # send likelihood observation to particle filter
    def __likelihood_obs_cb(msg):
      if len(msg.likelihood.nodes) == len(msg.likelihood.values):
        try:
          nodes = [np.where(self.node_names == nname)[0][0] for nname in msg.likelihood.nodes]
        except IndexError:
          self.logger.warn(
            "Received non-admissible node name {}, likelihood discarded".format(msg.likelihood.nodes))
        else:
          values = np.array(msg.likelihood.values)
          if np.isfinite(values).all() and (values >= 0.).all() and np.sum(values) > 0:
            p_estimated, particles = pf.receive_likelihood_obs(
              nodes, 
              msg.likelihood.values,
              # (rclpy.time.get_clock.now().to_sec(), msg.likelihood.header.stamp.to_sec())[
                # msg.likelihood.header.stamp.to_sec() > 0]
              rclpy.time.get_clock.now().to_sec(),
              identifying=msg.identifying
            )
            __publish(p_estimated.node, particles)
          else:
            self.logger.warn(
              "Received non-admissible likelihood observation {}, discarded".format(msg.likelihood.values))
      else:
        self.logger.warn("Nodes array and values array sizes do not match {} != {}, discarding likelihood observation".format(
          len(msg.likelihood.nodes), len(msg.likelihood.values)))

    # subscribe to topics receiving observation
    qos_profile = QoSProfile(depth=1, durability=QoSDurabilityPolicy.VOLATILE)
    self.obs_subscribers.append((
      self.create_subscription(PoseObservation, "{}/pose_obs".format(name), __pose_obs_cb, qos_profile),
      self.create_subscription(LikelihoodObservation, "{}/likelihood_obs".format(name), __likelihood_obs_cb, qos_profile)
    ))

    ## Services handlers ##
    # Get the pose observation and returns the localisation result
    def __update_pose_handler(request, response):
      if np.isfinite(request.pose.pose.pose.position.x) and \
         np.isfinite(request.pose.pose.pose.position.y) and \
         np.isfinite(request.pose.pose.covariance[0]) and \
         np.isfinite(request.pose.pose.covariance[7]):
        p_estimated, particles = pf.receive_pose_obs(
            request.pose.pose.pose.position.x,
            request.pose.pose.pose.position.y,
            request.pose.pose.covariance[0],  # variance of x
            request.pose.pose.covariance[7],  # variance of y
            # (rclpy.time.get_clock.now().to_sec(), request.pose.header.stamp.to_sec())[
            #     request.pose.header.stamp.to_sec() > 0]
            rclpy.time.get_clock.now().to_sec(),
            identifying=request.identifying
        )
        __publish(p_estimated.node, particles)
        response.success = True
        response.estimated_node = __prepare_cn_msg(p_estimated.node).data
        response.current_prob_dist = __prepare_pd_msg(particles)
        return(response)
      else:
        self.logger.warn(
          "Received non-admissible pose observation <{}, {}, {}, {}>, discarded".format(request.pose.pose.pose.position.x, request.pose.pose.pose.position.y, request.pose.pose.covariance[0], request.pose.pose.covariance[7]))
      
      # fallback negative response
      response.success = False
      return(response)

    # get a likelihood observation and return localisation result
    def __update_likelihood_handler(request, response):
      if len(request.likelihood.nodes) == len(request.likelihood.values):
        try:
          nodes = [np.where(self.node_names == nname)[0][0] for nname in request.likelihood.nodes]
        except IndexError:
          self.logger.warn("Received non-admissible node name {}, likelihood discarded".format(request.likelihood.nodes))
        else:
          values = np.array(request.likelihood.values)
          # self.logger.info("Received likelihood: {}".format(zip(nodes, values)))
          if np.isfinite(values).all() and (values >= 0.).all() and np.sum(values) > 0:
            p_estimated, particles = pf.receive_likelihood_obs(
              nodes,
              request.likelihood.values,
              # (rclpy.time.get_clock.now().to_sec(), request.likelihood.header.stamp.to_sec())[
                # request.likelihood.header.stamp.to_sec() > 0]
              rclpy.time.get_clock.now().to_sec(),
              identifying=request.identifying
            )
            __publish(p_estimated.node, particles)
            response.success = True
            response.estimated_node = __prepare_cn_msg(p_estimated.node).data
            response.current_prob_dist = __prepare_pd_msg(particles)
            return(response)
          else:
            self.logger.warn("Received non-admissible likelihood observation {}, discarded".format(request.likelihood.values))

      else:
        self.logger.warn("Nodes array and values array sizes do not match {} != {}, discarding likelihood observation".format(
          len(request.likelihood.nodes), len(request.likelihood.values)))

      # fallback negative response
      response.success = False
      return(response)

    def __do_stateless_prediction(request, response):
      # get a copy of the particle filter to work with
      _pf = pf.copy()
      # if requested pred rate is lesseq than 0 use the global one
      _prediction_rate = (request.prediction_rate, prediction_rate)[
        request.prediction_rate <= 0.]
      pred_step_secs = 1. / _prediction_rate
      secs_left = max(0.0, request.secs_from_now)
      time = rclpy.time.get_clock.now()
      response.success = True
      # sub-function to append predictions to the result message
      def ___append_prediction(node, particles, secs_passed, timestamp):
        if not (node is None or particles is None):
          __publish_stateless_viz(particles)
          response.secs_from_now.append(secs_passed)
          response.estimated_node.append(__prepare_cn_msg(node).data)
          response.prob_dist.append(__prepare_pd_msg(particles, timestamp=timestamp))
          return True
        else:
          response.success = False
          self.logger.warn(
            "Cannot perform prediction, no observation received so far.")
          return False
      # perform all the steps until reached limit time
      while secs_left > 0:
        p_estimated, particles = _pf.predict(
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
      p_estimated, particles = _pf.predict(
        timestamp_secs=time.to_sec()
      )
      _ = ___append_prediction(
            p_estimated.node, 
            particles,
            secs_passed=request.secs_from_now - secs_left,
            timestamp=time)

      return response

    def __do_stateless_update(request, response):
      # create a new PF and assign the prior distribution to start up with
      __pf = TopologicalParticleFilter(
        num=n_particles,
        prediction_model=pm,
        initial_spread_policy=initial_spread_policy,
        prediction_speed_decay=prediction_speed_decay,
        node_coords=self.node_coords,
        node_distances=self.node_distances,
        connected_nodes=self.connected_nodes,
        node_diffs2D=self.node_diffs2D,
        node_names=self.node_names
      )
      __pf.print_debug = False
      if len(request.likelihood.nodes) == len(request.likelihood.values) and \
          len(request.prior.nodes) == len(request.prior.values):
        try:
          lkl_nodes = [np.where(self.node_names == nname)[0][0] for nname in request.likelihood.nodes]
          pr_nodes = [np.where(self.node_names == nname)[0][0] for nname in request.prior.nodes]
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
            ts = rclpy.time.get_clock.now().to_sec()
            # send the prior (assignment bcs it's the first observation ever received)
            _, _ = __pf.receive_likelihood_obs(
              pr_nodes,
              request.prior.values,
              ts,
              True # this has to be true for initializing the distribution
            )
            # send the lkl (prediction is not performed here because the ts is the same as prior ts)
            p_estimated, particles = __pf.receive_likelihood_obs(
              lkl_nodes,
              request.likelihood.values,
              ts,
              False  # this has to be false to avoid the risk it gets re-initialised uniformly by the JSD threshold
            )
            response.success = True
            response.estimated_node = __prepare_cn_msg(p_estimated.node).data
            response.current_prob_dist = __prepare_pd_msg(particles)
            return(response)
          else:
            self.logger.warn(
              "Received non-admissible prior/likelihood observation {}, discarded".format(request.prior.values, request.likelihood.values))
      else:
        self.logger.warn("Nodes array and values array sizes do not match {} != {}/{} != {}, discarding prior/likelihood observation".format(
          len(request.prior.nodes), len(request.prior.values), len(request.likelihood.nodes), len(request.likelihood.values)))

    # subscribe to services for update
    self.upd_services.append({
      self.create_service(UpdatePoseObservation, "{}/update_pose_obs".format(name), __update_pose_handler),
      self.create_service(UpdateLikelihoodObservation, "{}/update_likelihood_obs".format(name), __update_likelihood_handler),
      self.create_service(Predict, "{}/predict_stateless".format(name), __do_stateless_prediction),
      self.create_service(UpdatePriorLikelihoodObservation, "{}/update_stateless".format(name), __do_stateless_update)
    })
    
    thr = None
    if do_prediction:
      # threaded function that performs predictions at a constant rate
      def __prediction_loop():
        rate = self.create_rate(prediction_rate)
        while not stop_event.is_set():
          p_estimate, particles = pf.predict(
            timestamp_secs=rclpy.time.get_clock.now().to_sec()
          )
          if not (p_estimate is None or particles is None): 
            __publish(p_estimate.node, particles)
          
          rate.sleep()
      
      thr = threading.Thread(target=__prediction_loop)
      thr.start()

    self.stopping_events.append(stop_event)
    self.prediction_threads.append(thr)
    self.agents.append(name)
    self.pfs.append(pf)

    self.logger.info("n_particles:{}, initial_spread_policy:{}, prediction_model:{}, do_prediction:{}, prediction_rate:{}. prediction_speed_decay:{}".format(
      n_particles, initial_spread_policy, prediction_model, do_prediction, prediction_rate, prediction_speed_decay
    ))

    # release resources
    self.internal_lock.release()

    self.logger.info("DONE")

    response.success = True
    return response

  def _stop_localise_handler(self, request, response):
    # """Remove specific localisation agent."""
    self.logger.info("Unregistering agent {} for localisation".format(request.name))
    self.internal_lock.acquire()
    # default name is unknown if requested is ''
    name = (request.name, 'unknown')[request.name == '']
    if name in self.agents:
      agent_idx = self.agents.index(name)
      # stop prediction loop
      if self.stopping_events[agent_idx] is not None:
        self.stopping_events[agent_idx].set()
      if self.prediction_threads[agent_idx] is not None:
        self.prediction_threads[agent_idx].join()

      # unregister topic subs
      for sub in self.obs_subscribers[agent_idx]:
        sub.unregister()
      # shutting down services
      for srv in self.upd_services[agent_idx]:
        srv.shutdown()
      # unregister topic pubs
      for pub in self.res_publishers[agent_idx]:
        pub.unregister()
      for pub in self.viz_publishers[agent_idx]:
        pub.unregister()

      # cleanup all the related variables
      del self.stopping_events[agent_idx]
      del self.prediction_threads[agent_idx]
      del self.obs_subscribers[agent_idx]
      del self.res_publishers[agent_idx]
      del self.viz_publishers[agent_idx]
      del self.upd_services[agent_idx]
      del self.agents[agent_idx]

      self.internal_lock.release()
      self.logger.info("DONE")
      response.success = True
      return response
    else:
      self.logger.warn("The agent {} is already not being localised.".format(name))
      self.internal_lock.release()
      response.success = False
      return response

  def _topo_map_cb(self, msg):
    """Receive the Topological Map."""
    self.topo_map = msg

    # save and compute topological map informations
    self.node_names = np.array([node.name for node in self.topo_map.nodes])
    self.node_coords = np.array(
      [[node.pose.position.x, node.pose.position.y] for node in self.topo_map.nodes])

    self.node_diffs2D = []
    self.node_distances = []
    self.connected_nodes = []
    for i, _ in enumerate(self.node_names):
      self.node_diffs2D.append(self.node_coords - self.node_coords[i])
      self.connected_nodes.append(
          np.array([np.where(self.node_names == edge.node)[0][0] for edge in self.topo_map.nodes[i].edges]))

    self.node_diffs2D = np.array(self.node_diffs2D)
    #self.connected_nodes = np.array(self.connected_nodes)
    
    self.node_distances = np.sqrt(np.sum(self.node_diffs2D ** 2, axis=2))
    self.logger.info("Received topomap")
    
  def close(self):
    # stop all the threads
    for thr, stop_event in zip(self.prediction_threads, self.stopping_events):
      stop_event.set()
      thr.join()


def main(args=None):
  rclpy.init(args=args)
  localisation_node = TopologicalLocalisation()
  rclpy.spin(localisation_node)
  rclpy.shutdown()


if __name__ == "__main__":
  main()
