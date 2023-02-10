# Bachelorarbeit

Project for the bachlor-thesis of **Mohamed Ghazal** from the **TU Kaiserslautern***

## Requirements
python 3.10 and its requirments are in the file requirements.txt

### First steps
Install **virtualenv** and create a new virtual environment fpr python.
```
virtualenv venv
```
and activate it using
```
source venv/bin/activate
```
Then install the required python packages with the command
```
pip3 install -r requirements
```
Templates and perepared examples can be found in the folder experiments. Copy the Experiement.json (depending on your needs change it, to perform other experiments) and labels.json files to the main directory and execute the command
```
pyhon3 src/experiment.py
```
To compile the docs the LateX engine XeLaTex must be installed on your machine and the compilation is done using the command after givin the file docs/compile.sh execution rights and changing to the folder docs
```
./compile.sh
```
