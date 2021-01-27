# Football player classifier

## Identification with API
### Setup API
In parent directory run
```bash
sudo bash run_docker_api.sh
```
### Make a query
Terminal
```bash
curl -X POST "http://127.0.0.1:5000/predict/" -H  "accept: application/json" -H  "Content-Type: multipart/form-data" -F "file=@path/to/your/image.png;type=image/png"
```
Or open GUI in your browser with following link  http://127.0.0.1:5000/docs

## Reuse data preprocessing and model training
If you want to train new model follow these steps.

First of all change to deep learning part's directory with
```bash
cd deep_learning_model
```
### Using virtual environment
Setup virtual environment by running
```bash
source create_venv.sh
```
To launch data preprocessing pipeline
```bash
jupyter-lab data_preprocessing.ipynb
```
To train new model
```bash
cd training
python3 train.py
```
To evaluate existing model:
```bash
cd training
python3 eval.py
```

### Using Docker
```bash
sudo bash run_docker_train.sh
```
To launch data preprocessing pipeline
```bash
jupyter notebook --ip 0.0.0.0 --no-browser --allow-root
```
To train new model
```bash
cd training
python3 train.py
```
To evaluate existing model:
```bash
cd training
python3 eval.py
```
After evaluation see ROC curve plot in directory and balanced accuracy in plot's name last value
