#FROM pytorch/pytorch:1.12.0-cuda11.3-cudnn8-runtime
FROM pytorch/pytorch:2.7.0-cuda11.8-cudnn9-runtime

# OpenCV
RUN apt update
RUN apt -y install ffmpeg libsm6 libxext6

# non-root user
RUN groupadd -r algorithm && useradd -m --no-log-init -r -g algorithm algorithm
RUN mkdir /input /opt/algorithm /output
RUN chown -R algorithm:algorithm /input /opt/algorithm /opt/conda /output
USER algorithm

# conda
RUN conda config --add channels conda-forge
RUN conda config --set channel_priority strict
RUN conda update -y conda

# pip requirements
WORKDIR /opt/algorithm
COPY --chown=algorithm:algorithm requirements.txt .
RUN python -m pip install --upgrade pip
RUN python -m pip install -r requirements.txt
RUN python -m pip install torch-scatter -f https://data.pyg.org/whl/torch-2.7.0+cu118.html

# copy package and main file
ENV LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:/opt/conda/lib/
COPY --chown=algorithm:algorithm jawfrac/ jawfrac/
COPY --chown=algorithm:algorithm process.py .

# copy checkpoints
#COPY --chown=algorithm:algorithm checkpoints/mandibles.ckpt checkpoints/mandibles.ckpt
#COPY --chown=algorithm:algorithm checkpoints/old_fractures_linear_displaced_patch_size=64.ckpt checkpoints/fractures.ckpt

# script to run
ENTRYPOINT ["python", "/opt/algorithm/process.py"]
