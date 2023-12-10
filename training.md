To start pretraining: run python pretraining_50.py.
It will save model named ce.pth which will be trained for 50 epochs with Cross Entropy Loss

Run python train_EDL500.py to start training on ce.pth with custom loss function i.e. EDLLoss() for 500 epochs and lr = 2e-2 
This will save model named unc500.pth

Run python cv_project/resume500_700.py to start training on unc500.pth with EDLLoss() for another 200 epochs with lr = 2e-5
This will save model named unc700.pth

Run python cv_project/resume700_900.py to start training on unc700.pth with EDLLoss() for another 200 epochs with lr = 1e-2
This will save model named unc900.pth
