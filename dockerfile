FROM tensorflow/tensorflow:2.15.0-gpu

# Install system packages
RUN apt-get update && apt-get install -y \
    build-essential \
    wget \
    curl \
    libtool \
    libffi-dev \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Install TA-Lib
COPY dependencies/ta-lib_0.6.4_amd64.deb ./dependencies/
RUN dpkg -i ./dependencies/ta-lib_0.6.4_amd64.deb || (apt-get update && apt-get install -f -y)

# Install Python packages
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Enable hidden files in Jupyter
RUN mkdir -p /root/.jupyter && \
    echo "c.ContentsManager.allow_hidden = True" >> /root/.jupyter/jupyter_notebook_config.py

# Expose and launch Jupyter
EXPOSE 8888
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--allow-root", \
     "--NotebookApp.token=", "--NotebookApp.password=", \
     "--NotebookApp.disable_check_xsrf=True", "--NotebookApp.allow_origin='*'", "--NotebookApp.allow_remote_access=True"]
