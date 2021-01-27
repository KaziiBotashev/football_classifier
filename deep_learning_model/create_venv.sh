#!/bin/bash
curl "https://www.dropbox.com/s/689m883ay99u9gx/soccerenet-epoch%3D37-val_loss%3D0.16.ckpt?dl=1" -O -J -L
curl "https://www.dropbox.com/s/czq9fmf6qz4roqr/resnet50_msmt17_combineall_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip_jitter.pth?dl=0" -O -J -L
mv soccerenet-epoch=37-val_loss=0.16.ckpt trained_model/
mv resnet50_msmt17_combineall_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip_jitter.pth trained_model/
python3 -m venv env
pip3 install --upgrade pip
source env/bin/activate
echo "Please wait..."
echo "Virtual environment activated" 
pip3 install -r requirements.txt
echo "All dependencies are installed"
