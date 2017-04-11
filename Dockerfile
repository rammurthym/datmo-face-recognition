FROM acusensehub/dlib-scikit:14.04

VOLUME ["/home/_data", "/home/_inputs", "/home/_outputs", "/home/src"]

# Setup environment variables
ENV INPUT_DIR=/home/_inputs
ENV OUTPUT_DIR=/home/_outputs
ENV DATA_DIR=/home/_data
ENV SRC_DIR=/home/src

# Run commands to make code work
RUN sudo apt-get update -y

# Numpy / Scipy reqs
RUN sudo apt-get install python-pandas -y
RUN sudo apt-get install python-sympy -y

RUN mkdir -p /home/src

COPY src /home/src

RUN find /home/src/scripts -name "*.sh" -exec chmod +x {} +
RUN pip install face_recognition

# Working directory: this is where unix scripts will run from
WORKDIR /home/src
