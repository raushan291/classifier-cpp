# Use a base image that includes the necessary runtime environment
# For Linux, you might use a Debian or Ubuntu image
# FROM ubuntu:latest

FROM python:3.10-slim

# install python
# RUN apt-get update
# RUN apt-get install -y python3.10 python3-pip

# Set the working directory
WORKDIR /app

# Copy the precompiled executable and shared library (if any) into the container
COPY Infer_OpenCV.exe /app/
COPY lib /app/lib/

# Copy necessary input/output/models directory
COPY models /app/models/

# Copy the Flask application
COPY app.py /app/
COPY templates /app/templates/

# Make required dirs
RUN mkdir -p uploads/

# Set the LD_LIBRARY_PATH environment variable to locate the shared library
ENV LD_LIBRARY_PATH=/app/lib/

# Install required packages
RUN pip install Flask
RUN pip install gevent 

# Expose port 5000 for the Flask application
EXPOSE 5000 5000

# Run the Flask application
CMD ["python", "app.py"]
