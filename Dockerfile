FROM python:3.9-slim

WORKDIR /app

# Install required packages
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy application files
COPY . .

# Expose port for Streamlit
EXPOSE 8080

# Set environment variables
ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8

# Run Streamlit
CMD ["streamlit", "run", "app.py", "--server.port=8080", "--server.address=0.0.0.0"]