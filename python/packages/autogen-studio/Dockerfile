FROM python:3.10-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    gcc \
    g++ \
    make \
    python3-dev \
    rsync \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Node.js and yarn
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && \
    apt-get update && \
    apt-get install -y nodejs && \
    npm install -g yarn gatsby-cli && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /code

# Create user
RUN useradd -m -u 1001 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    AUTOGENSTUDIO_APPDIR=/home/user/app

WORKDIR $HOME/app

# Copy source code
COPY --chown=user . $HOME/app

# Install Python dependencies from source
RUN pip install -e .
RUN pip install gunicorn uvicorn

# Build frontend
RUN cd frontend && \
    yarn install && \
    yarn build

# Expose the port
EXPOSE 8081

CMD ["autogenstudio", "ui", "--host", "0.0.0.0", "--port", "8081"]
