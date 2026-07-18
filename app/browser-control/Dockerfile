# Use the official Playwright Python image which includes necessary system dependencies
FROM mcr.microsoft.com/playwright/python:v1.40.0-jammy

# Set working directory
WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies globally (no virtual environment)
RUN pip install --no-cache-dir -r requirements.txt

# Install Playwright browsers (ensure Chromium is available)
RUN playwright install chromium

# Copy the rest of the application code
COPY . .

# Create directories for artifacts
RUN mkdir -p data browser_data screenshots archive debug_artifacts/dom_cache

# Expose the WebSocket/HTTP port
EXPOSE 8000

# Run the app wrapper (WebSocket server)
CMD ["python", "app_wrapper.py"]