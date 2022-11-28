#!/usr/bin/env bash

# install dependencies
conda install pytorch==1.6.0 torchvision==0.7.0 cudatoolkit=10.1 -c pytorch --yes
pip install tensorboard==2.4.1 --no-cache-dir
pip install tqdm==4.60.0 --no-cache-dir

# install mayavi for visualization
pip install vtk==9.1.0 --no-cache-dir
pip install pyQt5==5.15.2 --no-cache-dir
pip install mayavi==4.7.4 --no-cache-dir
