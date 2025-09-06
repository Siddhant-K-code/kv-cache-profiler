FROM mcr.microsoft.com/devcontainers/python:3.13-bullseye

# Install basic development tools
RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
    && apt-get -y install --no-install-recommends \
        build-essential \
        git \
        curl \
        wget \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
