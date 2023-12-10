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
To start pretraining: run python pretraining_50.py.
It will save model named ce.pth which will be trained for 50 epochs with Cross Entropy Loss

Run python train_EDL500.py to start training on ce.pth with custom loss function i.e. EDLLoss() for 500 epochs and lr = 2e-2 
This will save model named unc500.pth

Run python cv_project/resume500_700.py to start training on unc500.pth with EDLLoss() for another 200 epochs with lr = 2e-5
This will save model named unc700.pth

Run python cv_project/resume700_900.py to start training on unc700.pth with EDLLoss() for another 200 epochs with lr = 1e-2
This will save model named unc900.pth

run inference.py to compare your noising and denoising images uncertainty

run inference_custom.py to infer custom images stored in data/images/custom/ on model unc900.pth and store results in data/images/results/

run confusion_matrix.py to run model's inference on test images of flower 102 and perform uncertainty filtering to give out confusion matrices.

