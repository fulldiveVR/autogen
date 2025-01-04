FROM mcr.microsoft.com/devcontainers/python:3.10

# Install Node.js and yarn
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && \
    curl -sS https://dl.yarnpkg.com/debian/pubkey.gpg | apt-key add - && \
    echo "deb https://dl.yarnpkg.com/debian/ stable main" | tee /etc/apt/sources.list.d/yarn.list && \
    apt-get update && \
    apt-get install -y nodejs yarn && \
    npm install -g gatsby-cli

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
