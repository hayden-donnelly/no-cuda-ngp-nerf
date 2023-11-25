FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu20.04
RUN apt-get update -y \ 
    && apt-get install $NO_RECS -y \
        python3-dev=3.8.2-0ubuntu2 \
        python3-pip=20.0.2-5ubuntu1.9 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

ARG JAX_PACKAGE_URL="https://storage.googleapis.com/jax-releases/jax_cuda_releases.html"
ARG NO_CACHE="--no-cache-dir"
COPY requirements.txt requirements.txt
RUN python3 -m pip install $NO_CACHE --upgrade pip \
    # Install jax with GPU support.
    && python3 -m pip install $NO_CACHE \
        "jax[cuda11_cudnn86]" -f $JAX_PACKAGE_URL \
    # Jupyterlab is only required for the container.
    && python3 -m pip install $NO_CACHE \
        jupyterlab==4.0.5 \
    # Install other requirements.
    && python3 -m pip install $NO_CACHE \
        -r requirements.txt 

WORKDIR project
EXPOSE 7070
ENTRYPOINT ["jupyter", "lab", "--ip=0.0.0.0", "--allow-root", "--no-browser", "--NotebookApp.token=''", "--NotebookApp.password=''"]
