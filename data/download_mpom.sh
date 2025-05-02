#!/bin/bash
# 下载 MPOM 数据集脚本
DATA_DIR="$(dirname "$0")/mpom"
mkdir -p ${DATA_DIR}
# 示例下载命令，替换为真实链接
wget -O ${DATA_DIR}/mpom.zip http://example.com/mpom.zip
unzip ${DATA_DIR}/mpom.zip -d ${DATA_DIR}
