FROM ubuntu:latest

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    clang \
    git \
    cmake \
    && rm -rf /var/lib/apt/lists/*


RUN git clone https://github.com/KNGLJordan/strhive-to-fly.git .

WORKDIR /app/cpp

RUN mkdir -p build \
    && cd build \
    && cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_RUNTIME_OUTPUT_DIRECTORY=Release/ \
    && cmake --build . --config Release

WORKDIR /app

CMD ["./cpp/build/Release/strivecpp"]