from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
  return LaunchDescription([
    Node(
      package="bayesian_topological_localisation",
      executable="tpf_car_handler.py",
      name="tpf_car_handler"
    ),
    Node(
      package="bayesian_topological_localisation",
      executable="localisation_node.py"
    )
  ])
