# STEP 1: Install base image. Optimized for Python.
FROM python:3.7

# Step 2: Add requirements.txt file 
COPY requirements.txt /requirements.txt

# Step 3:  Install required pyhton dependencies from requirements file
RUN pip3 install -r requirements.txt
#RUN pip3 install requests

# Step 4: Copy source code in the current directory to the container
#ADD . /app
#MKDIR app
COPY model.py model.py

# Step 5: Set working directory to previously added app directory
WORKDIR /

# Step 6: Expose the port Flask is running on
EXPOSE 8000

# Step 9: Run Flask
ENV FLASK_APP model
CMD ["flask", "run", "--host=0.0.0.0", "--port=8000"]