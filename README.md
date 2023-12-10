Assuming Linux Operating system
Download the dataset zip folder from https://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz 
Place the downloaded folder in directory: cv_project/data/images/ 
Open terminal in directory cv_project/data/images/ and place the following command to extract these images: tar -xvzf 102flowers.tgz
Images will be saved in directory cv_project/data/images/jpg

make a new python env
conda create -n unc python=3.8

conda activate unc

run these commands after activating unc:

pip install torch==1.13.0+cu116 torchvision==0.14.0+cu116 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu116

pip install matplotlib
pip install pandas
pip install opencv-python
pip install scikit-image
pip install scipy


