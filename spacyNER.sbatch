#!/bin/bash
#SBATCH --job-name=spacy
#SBATCH --output=/home/%u/out/spacy.%j.out
#SBATCH --error=/home/%u/err/spacy.%j.err
#SBATCH -p normal
#SBATCH -c 1
#SBATCH --mem=32GB
ml python
pip3 install --user spacy
python3 -m spacy download en_core_web_sm
python3 spacyNER.py
