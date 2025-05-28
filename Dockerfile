# Use an official Python runtime as a parent image
FROM python:3.11.3-slim

# Set the working directory in the container to /app
WORKDIR /app

# Add the current directory contents into the container at /app
ADD . /app

ENV PROJECT_ID="440630564453" PORT=12725

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Make port 12725 available to the world outside this container
EXPOSE 12725

# Run the application when the container launches
CMD ["gunicorn", "--bind", ":12725", "--workers", "1", "--threads", "80", "--timeout", "0", "main:create_app()"]
