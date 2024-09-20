# Use an official Python runtime as the base image
FROM python:3.12-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY backend/requirements.txt .

# Install the required packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY backend/ .

# Copy the frontend build files
COPY frontend/build/ ./static/

# Expose the port that the app runs on
EXPOSE $PORT

# Define the command to run the application
CMD gunicorn --bind 0.0.0.0:$PORT app:app