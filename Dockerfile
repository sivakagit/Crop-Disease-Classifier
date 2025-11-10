# Use a TensorFlow-compatible Python base image
FROM python:3.10-slim

# Prevent interactive apt installs
ENV DEBIAN_FRONTEND=noninteractive

# Set working directory
WORKDIR /app

# Copy all project files into container
COPY . /app

# Install dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Expose port (Render expects 10000 internally, Flask defaults to 5000)
ENV PORT=5000
EXPOSE 5000

# Start the Flask app
CMD ["gunicorn", "backend.src.app:app"]
