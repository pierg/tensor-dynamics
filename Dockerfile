# Use an official NVIDIA image with TensorFlow, Python, and CUDA pre-installed
FROM nvcr.io/nvidia/tensorflow:23.09-tf2-py3

# Set environment variables to make Python output unbuffered, 
# which is useful when running Python within Docker containers
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Install system dependencies if needed
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    git \
    make && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Set the working directory within the container
WORKDIR /app

# Clone the repository inside the Docker container
RUN git clone https://github.com/pierg/neural_networks .

# Copy requirements file and install Python dependencies
# Note: This assumes that the requirements.txt file exists in the repository. 
# If it exists outside, you need to COPY it from your local context.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Expose port for TensorBoard
EXPOSE 6006

# Copy entrypoint script and grant execution permissions
COPY entrypoint.sh .
RUN chmod +x entrypoint.sh

# Use the entrypoint script to configure how the container will run
ENTRYPOINT ["/bin/bash", "entrypoint.sh"]

# The CMD defines default execution behavior for the container.
# It can be overridden by command-line parameters passed to `docker run`.
CMD []
