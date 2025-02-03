FROM continuumio/miniconda3:latest
WORKDIR /app

COPY environment.yml /app/

RUN conda env create -f environment.yml && \
    conda clean --all -f -y

COPY . /app

EXPOSE 8050

CMD ["conda", "run", "--no-capture-output", "-n", "dataflix", "python", "app.py"]









