FROM python:3.10

# Install system-level dependencies for TA-Lib
RUN apt-get update && apt-get full-upgrade -y && apt-get install -y \
    build-essential \
    wget \
    curl \
    libtool \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

# Add the TA-Lib .deb package and install it
COPY dependencies/ta-lib_0.6.4_amd64.deb ./dependencies/
RUN dpkg -i ./dependencies/ta-lib_0.6.4_amd64.deb || (apt-get update && apt-get install -f -y)

# Install Python packages
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

# Expose Jupyter Notebook
EXPOSE 8888

# Enable hidden files in JupyterLab
RUN mkdir -p /root/.jupyter && \
    echo "c.ContentsManager.allow_hidden = True" >> /root/.jupyter/jupyter_notebook_config.py


# Launch Jupyter
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--allow-root", "--NotebookApp.token=", "--NotebookApp.password=", "--NotebookApp.disable_check_xsrf=True", "--NotebookApp.allow_origin='*'", "--NotebookApp.allow_remote_access=True"]

