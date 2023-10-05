#!/bin/bash

clear
eval "$(conda shell.bash hook)"
conda activate timspeak
python -m unittest -v test_input
conda deactivate
