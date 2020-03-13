FROM ikeyasu/opengl:cuda9.0-cudnn7-devel-ubuntu16.04
MAINTAINER vkurenkov <v.kurenkov@innopolis.ru>


############################################
# Basic dependencies
############################################
RUN apt-get update --fix-missing && apt-get install -y \
      python3-numpy python3-matplotlib python3-dev \
      python3-opengl python3-pip \
      cmake zlib1g-dev libjpeg-dev xvfb libav-tools \
      xorg-dev libboost-all-dev libsdl2-dev swig \
      git wget openjdk-8-jdk ffmpeg unzip\
    && apt-get clean && rm -rf /var/cache/apt/archives/* /var/lib/apt/lists/*


############################################
# Change the working directory
############################################
WORKDIR /opt


############################################
# OpenAI Gym
############################################
RUN pip3 install --upgrade pip
RUN pip3 install h5py future pyvirtualdisplay 'gym[atari]' 'gym[box2d]' 'gym[classic_control]' opencv-python


############################################
# Deep Reinforcement Learning
#    OpenAI Baselines
############################################
# Need to remove mujoco dependency from baselines
RUN git clone --depth 1 https://github.com/openai/baselines.git \
    && sed --in-place 's/mujoco,//' baselines/setup.py \
    && pip3 install mpi4py cloudpickle


############################################
# PyTorch and Tensorflow
############################################
RUN pip3 install torch torchvision tensorflow-gpu==1.8


############################################
# PyBullet
############################################
RUN pip3 install pybullet==2.5.0
RUN sed -i 's/robot_bases/pybullet_envs.robot_bases/' /usr/local/lib/python3.5/dist-packages/pybullet_envs/robot_locomotors.py


############################################
# EvoRobotPy
############################################
RUN apt-get update && apt-get install libgsl0-dev -y
RUN pip3 install pyglet Cython
RUN git clone https://github.com/snolfi/evorobotpy.git

# EvoNet Setup
RUN cd evorobotpy/lib/ \
    && python3 setupevonet.py build_ext --inplace \
    && cp net*.so ../bin

# ErDiscrim Environment Setup
RUN cd evorobotpy/lib/ \
    && python3 setupErDiscrim.py build_ext --inplace \
    && cp ErDiscrim*.so ../bin

# ErPredPrey Environment Setup
RUN cd evorobotpy/lib/ \
    && python3 setupErPredprey.py build_ext --inplace \
    && cp ErPredprey*.so ../bin

# ErDpole Environment Setup
RUN cd evorobotpy/lib/ \
    && python3 setupErDpole.py build_ext --inplace \
    && cp ErDpole*.so ../bin


############################################
# Jupyter Notebook
############################################
RUN pip3 install notebook
