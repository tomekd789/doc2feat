# doc2feat

A proposal for extracting human readable features from large text corpuses.
The code is a reference to the following paper: http://viXra.org/abs/1703.0063

## Usage note:
After cloning the repository build your local copy of the executable by running make. Makefile is provided.
Then run the program on your dataset in e.g. the following way:

doc2feat -train ~/corpus -output ~/tag -size 500 -negative 10 -iter 10 -k-iter 10 -window 10 -words-per-feat 50 -sample 1e-4 -save-vocab ~/vocab -save-semantic-space ~/sem-space -features ~/feats -sparse 1
('-read' may be then replaced by '-save' on subsequent runs, to save on processing time).

After reviewing the generated feature file you may want to use the analyze-feature.py program to retrieve documents representing a feature of your interest.

This software is written for the linux operating system.
