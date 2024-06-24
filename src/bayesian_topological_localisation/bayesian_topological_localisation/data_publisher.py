import rclpy
from rclpy.node import Node

import numpy as np
from datetime import datetime
from astropy.time import Time as astrotime
import json
import pyproj

from bayesian_topological_localisation_msgs.msg import DistributionStamped, PoseObservation, LikelihoodObservation, ParticlesState
from bayesian_topological_localisation_msgs.srv import UpdatePoseObservation

class DataPublisher(Node):

  def __init__(self):
    super().__init__("data_pub")
    #self.pub_data = self.create_publisher(PoseObservation, "/bayesian_topological_localisation_agent_unknown/pose_obs", 10)
    self.proj = pyproj.Proj(proj="utm", zone=30, ellps="WGS84", units="m")
    self.map_zero = self.proj(-0.524509505881, 53.268642038)
    self.cli_pose_obs = self.create_client(srv_type=UpdatePoseObservation,
                                           srv_name="/bayesian_topological_localisation_agent_john/update_pose_obs")
    self.publish()
    return

  def publish(self):
    msg = PoseObservation()
    msg.pose.pose.pose.position

    # Get trolley data
    file = open("/home/niko/Dropbox/phd/topo_localisation/data/20240119/trolley.txt")
    trolley_ids = ["bcddc2cfcb68",  # static 1
                   "246f284a6c94",  # static 2
                   "70b8f606c710",  # mobile 1
                   "0cb8158460c0",  # mobile 2
                   "e831cd35d0f4"]  # mobile 3
    for line in file:
      line = json.loads(line)
      try:
        time = astrotime(datetime(int(line["YEAR"]),
                                  int(line["MONTH"]),
                                  int(line["DAY"]),
                                  int(line["HOUR"]),
                                  int(line["MINUTE"]),
                                  int(float(line["SECOND"])))).gps
        lat = float(line["LATITUDE"])
        lon = float(line["LONGITUDE"])
        alt = float(line["MSL_ALTITUDE"])
        fix = float(line["FIX_STATUS"])
        nos = float(line["GPS_SATELITES_USED"])
        hdop = float(line["HDOP"])
        vdop = float(line["VDOP"])
        pdop = float(line["PDOP"])

      except:
        continue

      if line["CLIENT_ID"] == trolley_ids[2]:
        print("lat: {:.6f}, lon: {:.6f}".format(lat, lon))
        trolley_loc = self.proj(lon, lat)
        lon = trolley_loc[0] - self.map_zero[0]
        lat = trolley_loc[1] - self.map_zero[1]

        req = UpdatePoseObservation.Request()
        req.pose.pose.pose.position.x = lon
        req.pose.pose.pose.position.y = lat
        req.pose.pose.pose.position.z = 0.2
        req.identifying = True
        print("x: {:.6f}, y: {:.6f}".format(lon, lat))
        future = self.cli_pose_obs.call_async(req)
        rclpy.spin_until_future_complete(self, future)


if __name__ == '__main__':
  rclpy.init(args=None)
  dp = DataPublisher()
  rclpy.spin(dp)
  rclpy.shutdown()