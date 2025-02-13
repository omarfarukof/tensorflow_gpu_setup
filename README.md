# Tensorflow GPU Setup

## Install TensorFlow on Manjaro

Step 1. Update Your System.

Before installing any new software, it’s a good practice to update your package database. This ensures that you’re installing the latest version of the software and that all dependencies are up to date.

To update the package database, run the following command in the terminal:

    sudo pacman -Syu

Step 2. Installing Necessary Development Tools.

TensorFlow requires certain development tools and libraries to be installed on your system. Install the GCC compiler and Python packages by running:

    sudo pacman -S base-devel python python-pip

Step 3. Setting Up a Virtual Environment using `uv`.

    uv add "tensorflow[and-cuda]"
    uv sync

Step 4. Installing CUDA and cuDNN (for GPU Support)

If you have a compatible NVIDIA GPU and want to leverage its power for accelerated TensorFlow computations, you need to install CUDA and cuDNN.

Installing CUDA Toolkit

First, add the CUDA repository to your system:

    sudo pacman -S cuda

Install the CUDA packages:

    sudo pacman -S cuda-toolkit

Installing cuDNN:

    sudo pacman -S cudnn

