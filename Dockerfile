#FROM nvidia/cuda
#FROM nvidia/cuda:10.0-cudnn7-runtime
#FROM nvidia/cuda:10.1-cudnn8-runtime
FROM nvidia/cuda:10.1-cudnn7-runtime

ARG PYTHON_VERSION=3.7
ARG CONDA_VERSION=3
ARG CONDA_PY_VERSION=4.5.11

RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    python3-pip python3-dev libsm6 curl \
    #wget gcc g++ make cmake libglib2.0-0 libxext6 libxrender-dev
    # bzip2 libopenblas-dev pbzip2 libgl1-mesa-glx && \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# CONDA INSTALLATION
ENV PATH /opt/conda/bin:$PATH

RUN curl https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
                    -o /opt/miniconda.sh
RUN  /bin/bash /opt/miniconda.sh -b -p /opt/conda && \
     rm /opt/miniconda.sh && \
         /opt/conda/bin/conda clean -tipsy && \
             ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
                 echo “. /opt/conda/etc/profile.d/conda.sh” >> ~/.bashrc && \
                     echo “conda activate base” >> ~/.bashrc


COPY environment.yml /opt
RUN conda env create -f /opt/environment.yml

RUN mkdir /opt/app
COPY train.py /opt/app
COPY autoencoder.py /opt/app
COPY entrypoint.sh /opt/app
RUN chmod +x /opt/app/entrypoint.sh

WORKDIR /opt/app
CMD '/opt/app/entrypoint.sh'

# Run example: sudo docker run --gpus device=0 -v /home/eldorado/datasets/DRealWorldSR/Train_x2/train_LR_bicubic_from_HR:/opt/bicubic -v /home/eldorado/datasets/DRealWorldSR/Train_x2/train_LR:/opt/train_LR -v /home/eldorado/dev/image_enhacement_autoencoders/autoencoder_x2_s/logs:/tmp/logs -v /home/eldorado/dev/image_enhacement_autoencoders/autoencoder_x2_s/checkpoints:/tmp/checkpoints --env TENSORBOARD_OUTPUT=/tmp/logs --env CHECKPOINTS_OUTPUT=/tmp/checkpoints --env MODEL=Autoencoder_x2_s --env X_ROOT=/opt/bicubic --env Y_ROOT=/opt/train_LR -it image_enhacement_autoencoder
