#!/bin/bash
echo "Football player identification API"
echo "Please wait..."
echo "Downloading models weights..."
FILE1=deep_learning_model/trained_model/resnet50_msmt17_combineall_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip_jitter.pth
FILE2=deep_learning_model/trained_model/soccernet-0.ckpt
FILE3=deep_learning_model/trained_model/soccernet-1.ckpt
FILE4=deep_learning_model/trained_model/soccernet-2.ckpt
FILE5=deep_learning_model/trained_model/soccernet-3.ckpt
if [ -f "$FILE1" ]; then
    echo "$FILE1 exists."
else 
    echo "$FILE1 does not exist."
    curl "https://www.dropbox.com/s/czq9fmf6qz4roqr/resnet50_msmt17_combineall_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip_jitter.pth?dl=1" -O -J -L
    mv resnet50_msmt17_combineall_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip_jitter.pth deep_learning_model/trained_model/
fi

if [ -f "$FILE2" ]; then
    echo "$FILE2 exists."
else 
    echo "$FILE2 does not exist."
    curl "https://www.dropbox.com/s/s7tf8zzfsp7ghxw/soccernet-0.ckpt?dl=1" -O -J -L
    mv soccernet-0.ckpt deep_learning_model/trained_model/
fi

if [ -f "$FILE3" ]; then
    echo "$FILE3 exists."
else 
    echo "$FILE3 does not exist."
    curl "https://www.dropbox.com/s/34o4xfzwl2ix12y/soccernet-1.ckpt?dl=1" -O -J -L
    mv soccernet-1.ckpt deep_learning_model/trained_model/
fi

if [ -f "$FILE4" ]; then
    echo "$FILE4 exists."
else 
    echo "$FILE4 does not exist."
    curl "https://www.dropbox.com/s/gqj6hstojec84eb/soccernet-2.ckpt?dl=1" -O -J -L
    mv soccernet-2.ckpt deep_learning_model/trained_model/
fi

if [ -f "$FILE5" ]; then
    echo "$FILE5 exists."
else 
    echo "$FILE5 does not exist."
    curl "https://www.dropbox.com/s/tqwnnzug64zj9o2/soccernet-3.ckpt?dl=1" -O -J -L
    mv soccernet-3.ckpt deep_learning_model/trained_model/
fi

echo "Building and Running Docker image..." 
echo $(docker build -t football_classifier_api .)
echo "Building Docker image is finished."
echo "Running Docker container..."
if [ ! "$(docker ps -q -f name=football_classifier_api)" ]; then
    if [ "$(docker ps -aq -f status=exited -f name=football_classifier_api)" ]; then
        # cleanup
        docker rm football_classifier_api
    fi
    # run your container
    docker run -p 5000:80 football_classifier_api
fi
