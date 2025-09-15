#!/bin/bash

conda create -n GuidingAttention python=3.10 -y
eval "$(conda shell.bash hook)"
conda activate GuidingAttention

pip install torch==2.5.1 torchvision --index-url https://download.pytorch.org/whl/cu124
pip install torch-adopt ezc3d tqdm torchmetrics argparse pillow  imgui[full]  PyOpenGL  PyOpenGL_accelerate  glfw omegaconf loguru tensorboard

git clone https://github.com/nghorbani/human_body_prior.git ./external/vposer
git clone https://github.com/ahmedosman/SUPR.git ./external/supr
git clone https://github.com/rodrigobdz/lrp.git ./external/lrp

cd ./external/supr
pip install .
cd ../..

cd ./external/vposer
python setup.py develop
cd ../..

cd ./external/lrp/lrp
#Remove sanity checks
sed -i '119s/^/#/' core.py
sed -i '120s/^/#/' core.py
cd ..
pip install .
cd ../..