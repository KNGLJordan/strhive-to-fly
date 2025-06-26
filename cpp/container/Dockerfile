FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    build-essential \
    clang \
    git \
    cmake \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN git clone https://github.com/KNGLJordan/strhive-to-fly.git .

WORKDIR /app/cpp

RUN chmod +x scripts/clean.sh && ./scripts/clean.sh

RUN chmod +x scripts/build-release.sh && ./scripts/build-release.sh

RUN ls -l build

RUN ls -l build/Release

CMD ["./build/Release/strivecpp"]

