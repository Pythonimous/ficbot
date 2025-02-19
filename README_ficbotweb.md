<h1><img alt="anime" title="anime" width="20px" src="https://raw.githubusercontent.com/Pythonimous/Pythonimous/main/assets/anime.png">  <a href="https://ficbotweb.com/">An Anime character generator</a>  <img alt="anime" title="anime" width="20px" src="https://raw.githubusercontent.com/Pythonimous/Pythonimous/main/assets/anime.png"></h1>
<img src="https://raw.githubusercontent.com/Pythonimous/Pythonimous/main/assets/gifs/namegen.gif" width="700" />

------
An AI-powered Fan Fiction Writing Assistant.

Ficbot is a machine learning-based system that provides various tools to make a good starting point for an aspiring writer when they need a character, but don't know where to start.

Here you will find the source code for my **Flask-based web client** for the main application. You can find out more about the main project in **this [repository](https://github.com/Pythonimous/ficbot)**, and **[here](https://www.kaggle.com/dataset/37798ba55fed88400b584cd0df4e784317eb7a6708e02fd5a650559fb4598353)** - about the dataset.
## Features
- Image -> Name generator

### Planned features
- Name generators (from Bio, Image + Bio)
- Bio generators (from Name, Image)
- Image generators (from nothing, Name, Bio)
- Anime filter (to turn yourself into OC!)
- Complete OC generator (Nothing -> Name, Bio, Image) :)

## Installation
I. Create and activate a virtual environment.

1. [Windows (w/o WSL)](https://mothergeo-py.readthedocs.io/en/latest/development/how-to/venv-win.html)
2. [Linux (from "Step 1. Install Python") / Windows (w/WSL)](https://www.liquidweb.com/kb/how-to-setup-a-python-virtual-environment-on-windows-10/)


II. Install project requirements from [requirements.txt](https://github.com/Pythonimous/ficbot/blob/main/requirements.txt)
```bash
pip3 install -r requirements.txt
```

## Launching the server
### Linux
In a virtual environment and in a project folder:
```bash
export FLASK_ENV=development
export FLASK_APP=ficbotweb
flask run
```
### Windows
In a virtual environment and in a project folder:
```bash
set FLASK_ENV=development
set FLASK_APP=ficbotweb
flask run
```
Go to the address in the terminal (i.e. http://127.0.0.1:5000/) and have fun!
