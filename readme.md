# SpaCy NER Workflow

This repo contains two simple files that execute spaCy's [parts of speech](https://spacy.io/usage/linguistic-features/#pos-tagging) and [named entity](https://spacy.io/usage/linguistic-features/#named-entities) taggers on a directory of 
.txt files.

## File Overview

The files consist of:

1. [spacyNER.py](spacyNER.py): Runs spaCy on a corpus, outputing .csv files with 
POS and NER for all the words in your corpus.
2. [spacyNER.sbatch](spacyNER.sbatch): Creates a batch job for spacyNER.py.

## Usage instructions

### Setting up and Connecting to Sherlock

1. Before we log onto Sherlock, let's make sure we're going to have everything we need there and move inputs/corpus onto Sherlock. For info on transferring data to Sherlock, see:
[https://www.sherlock.stanford.edu/docs/storage/data-transfer/](https://www.sherlock.stanford.edu/docs/storage/data-transfer/). [rsync](https://www.sherlock.stanford.edu/docs/storage/data-transfer/#rsync) is probably the best program for
this, but if you prefer another, go with that. For rsync, you'd use the command 
``` 
rsync -a ~/path/to/local/data yourSUNetid@login.sherlock.stanford.edu:/scratch/users/$USER/corpus/
```
You'll need to tweak the local path because I don't know where your files are located, but the remote path (after the ":") should work fine to get your corpus into scratch, a fast storage system where it's best to do file 
reading/writing.

2. Now we can log onto Sherlock using ssh in the Terminal program on Mac[^1]. with the syntax: 
```
ssh yourSUNetID@sherlock.stanford.edu
```
### File Management

3. Once you're on Sherlock, you'll want access to these files:
```bash
git clone https://github.com/bcritt1/pos_ner_python_spacy.git
```

This will create a directory in your home space on Sherlock called "pos_ner_python_spacy" with all the files in this repository.

Once you have the directory--you can ```ls``` to verify it's there--

4. Let's also make three directories for the outputs of our process:
```
mkdir out err /scratch/users/$USER/outputs
```
### Running Code

5. Now, let's move into our new directory
```
cd spacy
```
and submit our sbatch file to slurm, Sherlock's job scheduler: 
```
sbatch spacyNER.sbatch
```
You can watch your program run with
```
watch squeue -u $USER
```
When it finishes running, you should see your outputs as .csv and .json files in the outputs/ 
directory on scratch. This data can then be used as an input for other processes, or analyzed on its own.

## Code Explanation

The above walkthrough is designed to be as easy as possible to execute. If it works for you and you don't want to know about the code, you may not need to read this. If you want to know more, or need to tweak something, this section will 
help.

### spacyNER.sbatch

A batch script that gives [slurm](https://slurm.schedmd.com/pdfs/summary.pdf), Sherlock's job scheduler, instructions on how to set up for and run our python code. Everything that starts with a "#" are directives to slurm, and everything 
after are commands we're executing in the terminal on Sherlock.

```bash
#!/usr/bin/bash
#SBATCH --job-name=nltk					# gives the job a descriptive name that slurm will use
#SBATCH --output=/home/users/%u/out/nltk.%j.out		# the filepath slurm will use for output files. I've configured this so it automatically inserts variables for your username (%u) and the job name (%j) above.
#SBATCH --error=/home/users/%u/err/nltk.%j.err		# the filepath slurm will use for error files. I've configured this so it automatically inserts variables for your username (%u) and the job name (%j) above.
#SBATCH -p hns						# the partition slurm will use for the job. Here it is hns (humanities and sciences), but you can use other partions (sh_part to see which you can access)
#SBATCH -c 1						# number of cores to use. This should be 1 unless you've rewritten the code to run in parallel
#SBATCH --mem=32GB					# memory to use. 32GB should be plenty, but if you're getting a memory error, you can increase
module load python/3.9.0				# load the most recent version of python on Sherlock
pip3 install spacy					# load nltk for python
python3 -m spacy download en_core_web_sm		# download the language models
python3 spacyNER.py					# run the python script
```

### spacyNER.py

The python file contains pretty frequent in-line documentation, which you can check out using either ```cat spacyNER.py``` or ```nano spacyNER.py``` for more detail. As an outline, the script loads language models for spaCy, uses them on 
a 
corpus of files that I route it to automatically using environment variables (it reads your username from Sherlock and uses that to find your files) to perform word tokenization, parts of speech tagging, and named entity tagging 
progressively. Because the processes build on each other (one's output is the next's input), there aren't a ton of tweaks to be made. While some more extensive text cleaning is generally desirable in NLP, NER tagging tends to rely to 
some extent on things like capitalization and exact phrases which may be altered by lowercasing or lemmatization, respectively. If you 
have issues, contact [us](mailto:srcc-support@stanford.edu) and ask for Brad Rittenhouse.

 #### Notes

[^1]: The syntax would be the same if you use Terminal on Linux or Windows Subsystem for Linux [(WSL)](https://learn.microsoft.com/en-us/windows/wsl/install). Using other programs is possible, but documenting them here would be 
impossible. 
