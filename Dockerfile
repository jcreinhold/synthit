# Use official python runtime as a parent image
FROM python:3.6-stretch
MAINTAINER Jacob Reinhold, jacob.reinhold@jhu.edu

# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
ADD . /app

# 1) Install any needed packages specified in requirements.txt
# 2) Install ANTsPy which currently requires a specific path
# 3) Install this package into the container
# 4) Setup matplotlib to not pull in a GUI
RUN pip install --upgrade pip && \
    pip install --trusted-host pypi.python.org -r requirements.txt && \
    pip install https://github.com/ANTsX/ANTsPy/releases/download/v0.1.4/antspy-0.1.4-cp36-cp36m-linux_x86_64.whl && \
    python setup.py install && \
    echo "backend: agg" > matplotlibrc
