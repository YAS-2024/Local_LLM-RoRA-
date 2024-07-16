FROM  nvidia/cuda:12.1.0-cudnn8-devel-ubuntu20.04

ENV NOTO_DIR /usr/share/fonts/opentype/notosans
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
ENV PATH=/usr/local/cuda/bin:$PATH

RUN apt update \
    && apt install -y \
    wget \
    bzip2 \
    git \
    curl \
    unzip \
    file \
    xz-utils \
    sudo \
    python3 \
    python3-pip

RUN mkdir -p ${NOTO_DIR} &&\
  wget -q https://noto-website-2.storage.googleapis.com/pkgs/NotoSansCJKjp-hinted.zip -O noto.zip &&\
  unzip ./noto.zip -d ${NOTO_DIR}/ &&\
  chmod a+r ${NOTO_DIR}/NotoSans* &&\
  rm ./noto.zip

RUN apt-get autoremove -y && apt-get clean && \
  rm -rf /usr/local/src/*



  