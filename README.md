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
curl -X POST "http://127.0.0.1:5000/predict/?use_individual_models=true" -H  "accept: application/json" -H  "Content-Type: multipart/form-data" -F "file=@/path/to/your/image.png;type=image/png"
```
You need to specify which approach will be used for classification by setting the variable **use_individual_models** in your query. Set **true** to use a method based on three separate sequential models, or **false** to use one universal model.

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
python3 train.py <model_num>
```
To evaluate existing model:
```bash
cd training
python3 eval.py <model_num>
```
Here **<model_name>** is variable that used to specify which model to be trained.
Set **0** to train model that predicts 5 classes such as:

0 - blue team
1 - white team
2 - main referee
3 - side referee
4 - others

Set **1** to train model that predicts ID of the blue team players
Set **2** to train model that predicts ID of the white team players
Set **3** to train one universal model that predicts any class of 25 used



After evaluation see ROC curve plot in directory and balanced accuracy in plot's name last value.
