# Install base image. Optimized for Python.
FROM python:3.7

# Install required pyhton dependencies from requirements file
COPY requirements.txt /requirements.txt
RUN pip3 install -r requirements.txt

# Copy source code in the current directory to the container
COPY model.py model.py

# Set working directory to previously added app directory
WORKDIR /

# Expose the port Flask is running on
EXPOSE 8000

# Run Flask
ENV FLASK_APP model
CMD ["flask", "run", "--host=0.0.0.0", "--port=8000"]