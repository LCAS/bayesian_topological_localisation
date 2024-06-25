#!/usr/bin/env python3

import numpy as np

from bayesian_topological_localisation.particle import Particle
from bayesian_topological_localisation.prediction_model import PredictionModel

FOLLOW_OBS = 0      # use the distribution of the first observation
SPREAD_UNIFORM = 1  # equally distributed along all nodes
CLOSEST_NODE = 2    # assigned to closest node

# if the entropy of the current distribution is smaller than this threshold,
# stop jumping to close nodes that are unconnected
DEFAULT_UNCONNECTED_JUMP_THRESHOLD = 0.8
# if the Jensen-Shannon Distance btw prior and likelihood is greater than this threshold, 
# reinitialize particles with the likelihood AND restart jumping to close unconnected nodes
DEFAULT_REINIT_JSD_THRESHOLD = 0.90


class TopologicalParticleFilter():

  def __init__(self, n_of_ptcl, topo_map,
               reinit_jsd_threshold=DEFAULT_REINIT_JSD_THRESHOLD, unconnected_jump_threshold=DEFAULT_UNCONNECTED_JUMP_THRESHOLD):
    self.n_of_ptcl = n_of_ptcl
    self.topo_map = topo_map
    self.pm = PredictionModel(pred_type=PredictionModel.CTMC,
                              topo_map=self.topo_map)

    # current particles
    self.particles = [None] * self.n_of_ptcl
    # particles after prediction phase
    self.predicted_particles = [None] * self.n_of_ptcl
    # particles weight
    self.W = np.ones((self.n_of_ptcl))

    # time of last update
    self.time = [None] * self.n_of_ptcl
    # life time in current node
    self.life = np.zeros((self.n_of_ptcl))
    # last estimated node
    self.last_estimate = None
    # if to jump to only connected nodes 
    self.only_connected = False        

    self.reinit_jsd_threshold = reinit_jsd_threshold
    self.unconnected_jump_threshold = unconnected_jump_threshold

  def _expand_distribution(self, prob, nodes):
    if type(nodes) == np.ndarray:
      _nodes = nodes.tolist()
    else:
      _nodes = nodes
    new_prob = np.zeros(self.topo_map.node_coords().shape[0])
    for i in range(new_prob.shape[0]):
      try:
        _idx = _nodes.index(i)
      except:
        pass
      else:
        new_prob[i] = prob[_idx]

    return new_prob

  def _normalize(self, arr):
    arr = np.array(arr)
    row_sums = np.sum(arr)  # for 1d array
    if row_sums == 0:
      arr = np.ones(arr.shape)
      row_sums = np.sum(arr)
      #if self.print_debug:
      #    rospy.logwarn("Array to normalise is zero, resorting to uniform")
    arr = arr.astype(float) / row_sums
    return arr

  def _normal_pdf(self, mu_x, mu_y, cov_x, cov_y, nodes):
    mean = np.array([mu_x, mu_y])                 # center of gaussian
    cov_x = np.max([cov_x, 0.2])
    cov_y = np.max([cov_y, 0.2])
    cov_M = np.matrix([[cov_x, 0.], [0., cov_y]])    # cov matrix
    det_M = cov_x * cov_y                           # det cov matrix
    diffs2D = np.matrix(self.topo_map.node_coords()[nodes] - mean)
    up = np.exp(- 0.5 * (diffs2D * cov_M.I * diffs2D.T).diagonal())
    probs = np.array(up / np.sqrt((2*np.pi)**2 * det_M))
    return self._normalize(probs.reshape((-1)))

  def _initialize_wt_pose(self, obs_x, obs_y, cov_x, cov_y, timestamp_secs):
    nodes_prob = self._normal_pdf(obs_x, obs_y, cov_x, cov_y, range(self.topo_map.node_coords().shape[0]))
    nodes_prob = self._normalize(nodes_prob)
    _particles_nodes = np.random.choice(range(self.topo_map.node_coords().shape[0]), self.n_of_ptcl, p=nodes_prob)
    # sample velocity (x,y components) as gaussian sample with mean 0.0 and a covariance
    _particles_vels = np.random.normal(0.0, 0.05, (self.n_of_ptcl, 2))
    # sample time (seconds) as exponential sample 
    _particles_lifes = np.random.exponential(scale=1.0, size=self.n_of_ptcl)

    self.particles = [
      Particle(node, vel, life, timestamp_secs)
      for node, vel, life in zip(_particles_nodes, _particles_vels, _particles_lifes)
    ]
    
    for idx in range(len(self.particles)):
      self.predicted_particles[idx] = self.particles[idx].__copy__()
    self.time = np.ones((self.n_of_ptcl)) * timestamp_secs
    self.W = np.ones((self.n_of_ptcl))

  def _initialize_wt_likelihood(self, nodes, likelihood, timestamp_secs):
    probs = self._normalize(np.array(likelihood))
    _particles_nodes = np.random.choice(nodes, self.n_of_ptcl, p=probs)
    # sample velocity (x,y components) as gaussian sample with mean 0.0 and a covariance
    _particles_vels = np.random.normal(0.0, 0.05, (self.n_of_ptcl, 2))
    # sample time (seconds) as exponential sample
    _particles_lifes = np.random.uniform(high=1, size=self.n_of_ptcl)

    self.particles = [
      Particle(node, vel, life, timestamp_secs)
      for node, vel, life in zip(_particles_nodes, _particles_vels, _particles_lifes)
    ]

    for idx in range(len(self.particles)):
      self.predicted_particles[idx] = self.particles[idx].__copy__()
    self.time = np.ones((self.n_of_ptcl)) * timestamp_secs
    # self.life = np.zeros((self.n_of_ptcl))
    self.W = np.ones((self.n_of_ptcl))
    
  def _initialize_uniform(self, timestamp_secs):
    prob = 1.0 / self.n_of_ptcl
    self._initialize_wt_likelihood(
      np.arange(self.topo_map.node_coords().shape[0]), 
      np.ones((self.topo_map.node_coords().shape[0])) * prob,
      timestamp_secs
    )

  # if p is a prob dist
  def _compute_entropy(self, p):
    prob_p = p + 1e-5 # to avoid getting -inf for log(0)
    entropy = np.sum(prob_p * np.log2(prob_p))
    entropy = -entropy

    return entropy

  #  p and q are prob dist, returns the KL divergence
  def _compute_kl(self, p, q):
    prob_p = p + 1e-5 # to avoid getting -inf for log(0)
    prob_q = q + 1e-5 # to avoid getting -inf for log(0)

    kld = np.sum(prob_p * np.log2(prob_p / prob_q))

    return kld

  # distance between two distributions, symmetric
  # this is a value between 0 and 1
  def _compute_jensen_shannon_distance(self, p, q):
    m = (p + q) / 2

    divergence = (self._compute_kl(p, m) + self._compute_kl(q, m)) / 2

    distance = np.sqrt(divergence)

    return distance

  def _predict(self, timestamp_secs):

    for particle_idx in range(self.n_of_ptcl):

      p = self.particles[particle_idx]

      _new_p = self.pm.predict(particle=p, timestamp_secs=timestamp_secs)

      self.predicted_particles[particle_idx] = _new_p

  # weighting with normal distribution with var around the observation
  def _weight_pose(self, obs_x, obs_y, cov_x, cov_y, timestamp_secs, identifying):
    _nodes = np.array([p.node for p in self.predicted_particles])
    #_vels = np.array([p.vel for p in self.predicted_particles])
    idx_sort = np.argsort(_nodes)
    nodes, indices_start, counts = np.unique(
      _nodes[idx_sort], return_index=True, return_counts=True)
    indices_groups = np.split(idx_sort, indices_start[1:])
    nodes = nodes.tolist()

    # weight pose
    _all_nodes = np.arange(self.topo_map.node_coords().shape[0])
    prob_dist = self._normal_pdf(obs_x, obs_y, cov_x, cov_y, _all_nodes)

    self.W = np.zeros((self.n_of_ptcl))
    for _, (node, indices) in enumerate(zip(nodes, indices_groups)):
      self.W[indices] = prob_dist[node]

    # compute distributions distance
    js_distance = self._compute_jensen_shannon_distance(self._expand_distribution(self._normalize(counts), nodes), prob_dist)

    # it measn the particles are "disjoint" from this obs
    if identifying and js_distance > self.reinit_jsd_threshold:
      #if self.print_debug:
      #    rospy.logwarn("Reinitializing particles, JS distance between prior and likelihood {} is greater than {}".format(
      #        js_distance, self.reinit_jsd_threshold))
      self._initialize_wt_pose(obs_x, obs_y, cov_x, cov_y, timestamp_secs)
      self.only_connected = False # we are not really sure now anymore

  # weighting wih a given likelihood distribution
  def _weight_likelihood(self, nodes_dist, likelihood, timestamp_secs, identifying):
    _nodes = np.array([p.node for p in self.predicted_particles])
    idx_sort = np.argsort(_nodes)
    nodes, indices_start, counts = np.unique(
      _nodes[idx_sort], return_index=True, return_counts=True)
    indices_groups = np.split(idx_sort, indices_start[1:])
    nodes = nodes.tolist()

    self.W = np.zeros((self.n_of_ptcl))
    for _, (node, indices) in enumerate(zip(nodes, indices_groups)):
      if node in nodes_dist:
        self.W[indices] = likelihood[nodes_dist.index(node)]

    # compute distributions distance
    js_distance = self._compute_jensen_shannon_distance(
      self._expand_distribution(self._normalize(counts), nodes), self._expand_distribution(self._normalize(likelihood), nodes_dist))

    # it measn the particles are "disjoint" from this obs
    if identifying and js_distance > self.reinit_jsd_threshold:
      #if self.print_debug:
      #    rospy.logwarn("Reinitializing particles, JS distance between prior and likelihood {} is greater than {}".format(
      #        js_distance, self.reinit_jsd_threshold))
      self._initialize_wt_likelihood(nodes_dist, likelihood, timestamp_secs)
      self.only_connected = False # we are not really sure now anymore

  # produce the node estimate based on topological mass from particles and their weight
  def _estimate_node(self, use_weight=True):
    _nodes = [p.node for p in self.predicted_particles]
    nodes, indices_start, counts = np.unique(_nodes, return_index=True, return_counts=True)
    masses = []
    if use_weight:
      for (_, index_start, count) in zip(nodes, indices_start, counts):
        masses.append(self.W[index_start] * count)
    else:
      masses = counts

    self.last_estimate = self.predicted_particles[indices_start[np.argmax(masses)]]
    # if self.print_debug:
    #     print("Node estimate: {} {}".format(self.node_names[self.last_estimate.node], self.last_estimate))

  def _add_noise(self, particle):
    # noise to the node, bernoulli
    if np.random.random() < 0.001:
      if self.only_connected:
        closeby_nodes = self.topo_map.connected_nodes()[particle.node]
      else:
        closeby_nodes = np.where((self.topo_map.node_distances()[particle.node]<=3))[0]
      particle.node = np.random.choice(closeby_nodes)
    
    particle.vel += np.random.normal(0.0, 0.0005)
    particle.life = max(0, particle.life + np.random.uniform(low=-0.1, high=0.1))

  def _resample(self, use_weight=True):
    if use_weight:
      prob = self._normalize(self.W)
      particles_idxs = np.random.choice(
        np.arange(len(self.particles)), self.n_of_ptcl, p=prob)
    else:
      particles_idxs = np.arange(len(self.particles))
    
    for pi, idx in enumerate(particles_idxs):
      self.particles[pi] = self.predicted_particles[idx].__copy__()

    # add noise to the state of the new particles
    for p in self.particles:
      # if self.print_debug: print("clean", str(p))
      self._add_noise(p)
      # if self.print_debug: print("noisy", str(p))

    # compute entropy of the new distribution
    nodes, indices_start, counts = np.unique([p.node for p in self.particles], return_index=True, return_counts=True)
    p_entropy = self._compute_entropy(self._normalize(counts))
    # if self.print_debug: 
    #     rospy.loginfo("Final entropy : {} ({})".format(p_entropy, self.only_connected))

    if not self.only_connected and p_entropy < self.unconnected_jump_threshold:
      self.only_connected = True
      #if self.print_debug:
      #    rospy.logwarn("Stop jumping to unconnected nodes, entropy of current particles distribution {} smaller than {}.".format(p_entropy, self.unconnected_jump_threshold))

  def set_JSD_upper_bound(self, bound):
    self.reinit_jsd_threshold = bound

  def set_entropy_lower_bound(self, bound):
    self.unconnected_jump_threshold = bound

  def predict(self, timestamp_secs):
    """Performs a prediction step, estimates the new node and resamples the particles based on the prediction model only."""

    if self.last_estimate is None: # never received an observation
      particles = None
      p_estimate = None
    else:
      self._predict(timestamp_secs)

      self._estimate_node(use_weight=False)

      self._resample(use_weight=False)

      particles = [None] * len(self.particles)
      for idx in range(len(self.particles)):
        particles[idx] = self.particles[idx].__copy__()
      p_estimate = self.last_estimate.__copy__()

    return p_estimate, particles

  def receive_pose_obs(self, obsx, obsy, covx, covy, timestamp_secs, identifying):
    """Performs a full bayesian optimization step of the particles by integrating the new pose observation"""

    if self.last_estimate is None:  # never received an observation before
      # if the observation can be a false positive do not initialize the PF with it
      if identifying:
        self._initialize_wt_pose(obsx, obsy, covx, covy, timestamp_secs)
      else:
        self._initialize_uniform(timestamp_secs)
      
      use_weight = False
    else:
      self._predict(timestamp_secs)

      self._weight_pose(obsx, obsy, covx, covy, timestamp_secs, identifying)
      
      use_weight = True

    self._estimate_node(use_weight=use_weight)

    self._resample(use_weight=use_weight)

    particles = [None] * len(self.particles)
    for idx in range(len(self.particles)):
      particles[idx] = self.particles[idx].__copy__()
    p_estimate = self.last_estimate.__copy__()

    return p_estimate, particles

  def receive_likelihood_obs(self, nodes, likelihood, timestamp_secs, identifying):
    """Performs a full bayesian optimization step of the particles by integrating the new likelihood distribution observation"""
    
    if self.last_estimate is None:  # never received an observation before
      # if the observation can be a false positive do not initialize the PF with it
      if identifying:
        self._initialize_wt_likelihood(
          nodes, likelihood, timestamp_secs)
      else:
        self._initialize_uniform(timestamp_secs)
      
      use_weight = False
    else:
      #self._update_speed()

      self._predict(timestamp_secs)

      self._weight_likelihood(
        nodes, likelihood, timestamp_secs, identifying)

      use_weight = True

    self._estimate_node(use_weight=use_weight)

    self._resample(use_weight=use_weight)

    particles = [None] * len(self.particles)
    for idx in range(len(self.particles)):
      particles[idx] = self.particles[idx].__copy__()
    p_estimate = self.last_estimate.__copy__()

    return p_estimate, particles

  def reinitialize_particles(self, timestamp_secs):
    self._initialize_uniform(timestamp_secs)

  def copy(self):
    """Factory function that produces a copy of the current object"""
    # create a new PF object
    copy_obj = TopologicalParticleFilter(num=self.n_of_ptcl,
                                         topo_map=self.topo_map)

    # get all the class variables
    variables = [attr for attr in dir(copy_obj) if not callable(
      getattr(copy_obj, attr)) and not attr.startswith("__")]
    
    # copy all the variable values, excluding some
    exclude_variables = ["lock", "particles", "predicted_particles"]
    for var in variables:
      if var in exclude_variables:
        continue

      if isinstance(getattr(self, var), np.ndarray):
        setattr(copy_obj, var, np.copy(getattr(self, var)))
      else:
        setattr(copy_obj, var, getattr(self, var))

    # copy the particles
    for idx in range(len(self.particles)):
      copy_obj.predicted_particles[idx] = self.predicted_particles[idx].__copy__()
      copy_obj.particles[idx] = self.particles[idx].__copy__()

    return copy_obj
