# Bachelorarbeit

Project for the bachlor-thesis of **Mohamed Ghazal** from the **TU Kaiserslautern***

## Requirements
python 3.9 and its requirments are in the file requirements.txt

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
To Launsh the simulation entre the following command and follow the instructions
```
python3 src/main.py
```

As for the urdf-tools for compiling and visualising urdf-Models the following software is needed
#### For Mac os:
- urdfdom 
- urdfdom_headers
To check the urdf-file run the following command:
```
check_urdf <PATH-TO-FILE>
```
To show a graphical representation of the urdf-file run the following command:
```
urdf_to_graphviz <PATH-TO-FILE>
```
The output of this command is two files:
- <NAME-OF-ROBOT>.pdf a graphical repersetation of the urdf-file as a directed-graph
- <NAME-OF-ROBOT>.gv that can be opened with graphviz with the following command, which saves the graph to the output-file **output.svg**
```
cat <NAME-OF-ROBOT>.gv | dot -Tsvg -o output.svg
```
