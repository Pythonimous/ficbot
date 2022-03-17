# Ficbot

An AI-powered Fan Fiction Writing Assistant.

Ficbot is a machine learning-based system that provides various tools to make a good starting point for an aspiring writer when they need a character, but don't know where to start.

The main driving force behind this project was the availability of similarly structured anime character data on websites like [MyAnimeList](https://myanimelist.net/).

Moreover, despite different anime studios having different artistic styles, the anime style itself allows character images to make a roughly homogenous dataset, which can be used to help different generation tasks (maybe, even train a GAN? :)

You may experiment with the models using terminal, or demo notebooks from **this [folder](https://github.com/Pythonimous/ficbot/tree/main/notebooks)**.
## Features
- Image -> Name generator

### Planned features
- Name generators (from Bio, Image + Bio)
- Bio generators (from Name, Image)
- Image generators (from nothing, Name, Bio)
- Anime filter (to turn yourself into OC!)
- Complete OC generator (Nothing -> Name, Bio, Image) :)


## Installation

Install project requirements from [requirements.txt](https://github.com/Pythonimous/ficbot/blob/main/requirements.txt)
```bash
pip3 install -r requirements.txt
```

## Usage example
### Simple Image -> Name generation
```bash
python3 main.py --framework tf --image_path example/name/example.jpg --model_path example/name/tf_simple_average.hdf5 --maps_path example/name/maps.pkl --min_name_length 2 --diversity 1.0
```
### Selecting model by inputs and outputs
```bash
python3 main.py --framework tf --inputs image --outputs name
```
### Train model from scratch
```bash
python3 main.py --train --framework tf --model simple_img_name --data_path data/interim/img_name.csv --name_col eng_name --img_col image --img_dir data/raw/images --checkpoint_dir checkpoints --batch_size 16 --epochs 5 --maxlen 3 --optimizer adam
```
### Train model from checkpoint
```bash
python3 main.py --train --framework tf --model simple_img_name --checkpoint example/name/tf_simple_average.hdf5 --maps example/name/maps.pkl --data_path data/interim/img_name.csv --name_col eng_name --img_col image --img_dir data/raw/images --checkpoint_dir checkpoints --batch_size 16 --epochs 5 --maxlen 3
```

## Dataset
The original dataset has been crawled from MyAnimeList.net using Selenium and publicly available Python wrapper for [Jikan API](https://jikan.moe/).

Raw version of the dataset available [**here**](http://www.kaggle.com/dataset/37798ba55fed88400b584cd0df4e784317eb7a6708e02fd5a650559fb4598353). You can redownload it using the [download.py](https://github.com/Pythonimous/ficbot/blob/main/ficbot/data/download.py) script.

This script requires character links to download character data. You can provide them yourself, or download them using the same script (use --help for more details)

## Testing
In order to confirm that the scripts are functional after your requirements are installed, use:
```bash
python3 -m unittest
```
Test coverage is not complete. You can check test coverage using [**coverage**](https://coverage.readthedocs.io/en/6.3.2/) library for Python. You can install it via
```bash
pip3 install coverage
```
Before generating the report, you need to run the tests using coverage:
```bash
coverage run -m unittest
```
A simple report can then be generated using:
```bash
coverage report
```