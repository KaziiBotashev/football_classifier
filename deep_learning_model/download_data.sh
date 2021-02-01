#!/bin/bash
echo "Please wait..."
echo "Downolading weights and data..."
FILE1=trained_model/resnet50_msmt17_combineall_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip_jitter.pth
FILE2=trained_model/soccernet-0.ckpt
FILE3=trained_model/soccernet-1.ckpt
FILE4=trained_model/soccernet-2.ckpt
FILE5=trained_model/soccernet-3.ckpt
DIR1=data/images_splited_balanced_upscaled_team0
DIR2=data/images_splited_balanced_upscaled_team1
DIR3=data/images_splited_balanced_upscaled_team2
DIR4=data/images_splited_balanced_upscaled_team3
if [ -f "$FILE1" ]; then
    echo "$FILE1 exists."
else 
    echo "$FILE1 does not exist."
    curl "https://www.dropbox.com/s/czq9fmf6qz4roqr/resnet50_msmt17_combineall_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip_jitter.pth?dl=1" -O -J -L
    mv resnet50_msmt17_combineall_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip_jitter.pth trained_model/
fi

if [ -f "$FILE2" ]; then
    echo "$FILE2 exists."
else 
    echo "$FILE2 does not exist."
    curl "https://www.dropbox.com/s/s7tf8zzfsp7ghxw/soccernet-0.ckpt?dl=1" -O -J -L
    mv soccernet-0.ckpt trained_model/
fi

if [ -f "$FILE3" ]; then
    echo "$FILE3 exists."
else 
    echo "$FILE3 does not exist."
    curl "https://www.dropbox.com/s/34o4xfzwl2ix12y/soccernet-1.ckpt?dl=1" -O -J -L
    mv soccernet-1.ckpt trained_model/
fi

if [ -f "$FILE4" ]; then
    echo "$FILE4 exists."
else 
    echo "$FILE4 does not exist."
    curl "https://www.dropbox.com/s/gqj6hstojec84eb/soccernet-2.ckpt?dl=1" -O -J -L
    mv soccernet-2.ckpt trained_model/
fi

if [ -f "$FILE5" ]; then
    echo "$FILE5 exists."
else 
    echo "$FILE5 does not exist."
    curl "https://www.dropbox.com/s/tqwnnzug64zj9o2/soccernet-3.ckpt?dl=1" -O -J -L
    mv soccernet-3.ckpt trained_model/
fi





if [ -d "$DIR1" ]; then
    echo "$DIR1 exists."
else 
    echo "$DIR1 does not exist."
    curl "https://www.dropbox.com/s/shje8r34kuzpa0p/images_splited_balanced_upscaled_team0.zip?dl=1" -O -J -L
    unzip images_splited_balanced_upscaled_team0.zip -d data/
    rm images_splited_balanced_upscaled_team0.zip
fi

if [ -d "$DIR2" ]; then
    echo "$DIR2 exists."
else 
    echo "$DIR2 does not exist."
    curl "https://www.dropbox.com/s/3psahsvkso3ak2b/images_splited_balanced_upscaled_team1.zip?dl=1" -O -J -L
    unzip images_splited_balanced_upscaled_team1.zip -d data/
    rm images_splited_balanced_upscaled_team1.zip
fi

if [ -d "$DIR13" ]; then
    echo "$DIR3 exists."
else 
    echo "$DIR3 does not exist."
    curl "https://www.dropbox.com/s/fr5lm3khun7o1gg/images_splited_balanced_upscaled_team2.zip?dl=1" -O -J -L
    unzip images_splited_balanced_upscaled_team2.zip -d data/
    rm images_splited_balanced_upscaled_team2.zip
fi

if [ -d "$DIR4" ]; then
    echo "$DIR4 exists."
else 
    echo "$DIR4 does not exist."
    curl "https://www.dropbox.com/s/en2sr213v3bnfwo/images_splited_balanced_upscaled_team3.zip?dl=3" -O -J -L
    unzip images_splited_balanced_upscaled_team3.zip -d data/
    rm images_splited_balanced_upscaled_team3.zip
fi

