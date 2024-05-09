FROM lcas.lincoln.ac.uk/lcas/ros:humble as base
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -qq -y --no-install-recommends \
    git \
    python3-pip \
    python3-rosdep \
    ros-humble-rmw-cyclonedds-cpp \
    ros-humble-cyclonedds

# get the source tree and analyse it for its package.xml only
FROM base as sourcefilter
COPY ./src /tmp/src
# remove everything that isn't package.xml
RUN find /tmp/src -type f \! -name "package.xml" -print | xargs rm -rf

# install all dependencies listed in the package.xml
FROM base as depbuilder
# copy the reduced source tree (only package.xml) from previous stage
COPY --from=sourcefilter /tmp/src /tmp/src

RUN rosdep update && apt-get update
RUN cd /tmp/src && rosdep install --from-paths . --ignore-src -r -y && cd && rm -rf /tmp/src

FROM depbuilder as dev_base
# add user ros with sudo rights
RUN useradd -ms /bin/bash ros && echo "ros:ros" | chpasswd && adduser ros sudo

# add sudo without password
RUN echo "ALL ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers

RUN echo "source /opt/ros/humble/setup.bash" >> /etc/bash.bashrc

FROM dev_base as build

COPY ./src /home/ros/ros_ws/src
RUN chown -R ros:ros /home/ros/ros_ws

USER ros
WORKDIR /home/ros/ros_ws

RUN /bin/bash -c "source /opt/ros/humble/setup.bash; colcon build"

CMD [ "bash", "-c", "source /home/ros/ros_ws/install/setup.bash; ros2 run bayesian_topological_localisation localisation_node.py" ]
