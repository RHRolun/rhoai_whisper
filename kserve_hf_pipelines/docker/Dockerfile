#FROM registry.access.redhat.com/ubi9/python-311:1-62
#using Centos 9 because I can't install ffmpeg on UBI
FROM quay.io/sclorg/python-311-c9s:c9s

# By default, listen on port 8080
EXPOSE 8080/tcp
ENV FLASK_PORT=8080

USER root
RUN yum install -y epel-release
RUN dnf install -y --nogpgcheck https://mirrors.rpmfusion.org/free/el/rpmfusion-free-release-$(rpm -E %rhel).noarch.rpm
RUN dnf install -y ffmpeg
USER 1001

# Set the working directory in the container
WORKDIR /projects

# Copy the content of the local src directory to the working directory
COPY . .

# Install any dependencies
RUN \
  if [ -f requirements.txt ]; \
    then pip install -r requirements.txt; \
  elif [ `ls -1q *.txt | wc -l` == 1 ]; \
    then pip install -r *.txt; \
  fi

# Specify the command to run on container start
ENTRYPOINT ["python", "./model.py"]