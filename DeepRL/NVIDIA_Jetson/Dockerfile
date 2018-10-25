FROM ubuntu

WORKDIR /.

# Install some basic utilities
RUN apt-get update && apt-get install -y \
    curl wget ca-certificates sudo g++ cmake git zip unzip

# Clone repository
RUN git clone http://github.com/dusty-nv/jetson-reinforcement \
  && cd jetson-reinforcement \
  && git submodule update --init \
  && mkdir build \
  && cd build \
  && -y cmake ../ \
  && -y cmake

EXPOSE 8888
