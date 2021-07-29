FROM continuumio/miniconda3

WORKDIR /app

# copy environment specification and requirements and create the env
COPY environment.yml requirements.txt requirements-dev.txt ./
RUN conda env create -f environment.yml

# activate the env automatically for each new shell
RUN echo "source activate encomp-env" > ~/.bashrc
ENV PATH /opt/conda/envs/encomp-env/bin:$PATH

# install coolprop
# TODO: remove this once coolprop supports Python 3.9
RUN conda install conda-forge::coolprop

# install latest version from pip
RUN pip install encomp

# test the installation
RUN pytest --pyargs encomp

# run the docker container with
# docker run -it encomp
