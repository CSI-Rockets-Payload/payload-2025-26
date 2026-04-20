FROM python:3.10

WORKDIR /app

COPY model/cnn1d.py .

# Install requirements
COPY requirements.txt /workspace/requirements.txt
RUN pip install --no-cache-dir -r /workspace/requirements.txt
RUN pip install torch==2.4.0 torchvision --index-url https://download.pytorch.org/whl/cpu


# Set working directory
WORKDIR /workspace

# Ensure Python can import from this path
#ENV PYTHONPATH=/workspace/src
ENV PYTHONPATH=/workspace:/workspace/src

CMD ["bash"]

#CMD ["python", "cnn1d.py"]


