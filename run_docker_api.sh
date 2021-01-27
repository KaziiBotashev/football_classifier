#!/bin/bash
echo "Football player identification API"
echo "Please wait..."
echo "Building and Running Docker image..." 
curl "https://www.dropbox.com/s/689m883ay99u9gx/soccerenet-epoch%3D37-val_loss%3D0.16.ckpt?dl=1" -O -J -L
curl "https://www.dropbox.com/s/czq9fmf6qz4roqr/resnet50_msmt17_combineall_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip_jitter.pth?dl=0" -O -J -L
mv soccerenet-epoch=37-val_loss=0.16.ckpt deep_learning_model/trained_model/
mv resnet50_msmt17_combineall_256x128_amsgrad_ep150_stp60_lr0.0015_b64_fb10_softmax_labelsmooth_flip_jitter.pth deep_learning_model/trained_model/
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
