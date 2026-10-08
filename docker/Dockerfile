ARG UBUNTU_VERSION=24.04

FROM ubuntu:$UBUNTU_VERSION AS build

RUN apt-get update && apt-get install -y --no-install-recommends build-essential git cmake ca-certificates && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /sd.cpp

COPY . .

RUN cmake . -B ./build \
    -DSD_BUILD_SHARED_LIBS=ON \
    -DGGML_NATIVE=OFF \
    -DSD_BUILD_SHARED_GGML_LIB=ON \
    -DGGML_BACKEND_DL=ON \
    -DGGML_CPU_ALL_VARIANTS=ON \
    -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON \
    -DCMAKE_INSTALL_RPATH='$ORIGIN'
RUN cmake --build ./build --config Release --parallel

FROM ubuntu:$UBUNTU_VERSION AS runtime

RUN apt-get update && \
    apt-get install --yes --no-install-recommends libgomp1 && \
    apt-get clean

COPY --from=build /sd.cpp/build/bin /sd.cpp/bin
RUN printf '#!/bin/sh\nexec /sd.cpp/bin/sd-cli "$@"\n' > /sd-cli && \
    printf '#!/bin/sh\nexec /sd.cpp/bin/sd-server "$@"\n' > /sd-server && \
    chmod +x /sd-cli /sd-server

WORKDIR /sd.cpp

ENTRYPOINT [ "/sd-cli" ]
