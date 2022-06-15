FROM nvidia/cuda:11.3.1-cudnn8-devel-ubuntu20.04

ADD . /LEPISZCZE

RUN apt-get update \
    && DEBIAN_FRONTEND="noninteractive" apt-get -y install tzdata \
    && apt-get install -y $(awk '{print $1}' docker/apt.packages) \
    && apt-get install -y zsh \
    && chsh -s /bin/zsh \
    && zsh -c "$(wget https://raw.github.com/robbyrussell/oh-my-zsh/master/tools/install.sh -O -)"

SHELL ["/bin/zsh", "-c"]

RUN git clone https://github.com/Syllo/nvtop.git \
    && mkdir -p nvtop/build \
    && cd nvtop/build  \
    && cmake .. -DNVIDIA_SUPPORT=ON -DAMDGPU_SUPPORT=ON \
    && make \
    && make install

ENV PATH="/root/miniconda3/bin:${PATH}"
ARG PATH="/root/miniconda3/bin:${PATH}"

RUN wget \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
    && mkdir ~/.conda \
    && bash Miniconda3-latest-Linux-x86_64.sh -b \
    && rm -f Miniconda3-latest-Linux-x86_64.sh

WORKDIR /LEPISZCZE

RUN conda init zsh

RUN source ~/.zshrc  \
    && conda create -n LEPISZCZE python=3.9 \
    && conda activate LEPISZCZE \
    && python -m pip install poetry

RUN source ~/.zshrc  \
    && conda activate LEPISZCZE \
    && poetry install \
    && poetry run poe force-torch-cuda

ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8

CMD exec /bin/zsh -c "trap : TERM INT; sleep infinity & wait"
