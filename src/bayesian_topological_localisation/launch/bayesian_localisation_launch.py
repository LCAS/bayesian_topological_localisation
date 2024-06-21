from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
  return LaunchDescription([
    Node(
      package="topological_navigation",
      executable="map_manager2.py",
      arguments=["/home/ros/ros_ws/src/bayesian_topological_localisation/config/riseholme_tagged.tmap2.yaml"],
      name="topological_map_manager"
    ),
    Node(
      package="topological_navigation",
      executable="topological_transform_publisher.py",
      name="topological_transform_publisher"
    ),
    Node(
      package="topological_navigation",
      executable="topomap_marker2.py",
      name="topological_map_visualiser"
    ),
    Node(
      package="bayesian_topological_localisation",
      executable="tpf_car_handler.py",
      name="tpf_car_handler"
    ),
    Node(
      package="bayesian_topological_localisation",
      executable="localisation_node.py",
      name="bayes_topological_localisation"
    )
  ])
