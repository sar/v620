# Add ARG to accept the base image dynamically
ARG BASE_IMAGE=rocm-base
FROM ${BASE_IMAGE}

# SET workdir
WORKDIR /app

# 1 Clone source code repository
# Beta branch
RUN git clone --depth 1 -b beta https://github.com/Kaden-Schutt/hipfire.git hipfire

# 2 Copy scripts
COPY install_hipfire.sh /app/install_hipfire.sh
COPY configs/ /app/configs/

# 3 Run build script
RUN ls -la /app/ && \
    chmod +x /app/*.sh && \
    /app/install_hipfire.sh

# 4 Set PATH for hipfire binaries
ENV PATH="/root/.hipfire/bin:${PATH}"

CMD ["/bin/bash"]
