FROM quay.io/modh/odh-pytorch-notebook@sha256:aaf5efdeb965abd7bb0fb330a3f63b58d86dd8bfd770335a371d296cb6ba50e2

# By default, listen on port 8080
EXPOSE 8080/tcp
ENV FLASK_PORT=8080

# User Root
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
ENTRYPOINT ["python", "./model_filename_input.py"]
