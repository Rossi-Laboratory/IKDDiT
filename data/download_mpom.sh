#!/bin/bash
# Download MPOM dataset script
DATA_DIR="$(dirname "$0")/mpom"
mkdir -p ${DATA_DIR}
# Sample download command, replace with real link
wget -O ${DATA_DIR}/mpom.zip http://example.com/mpom.zip
unzip ${DATA_DIR}/mpom.zip -d ${DATA_DIR}
