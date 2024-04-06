# BERT-based Idiom Identification using Language Translation and Word Cohesion

This repository is intended to provide a collection of the most important files related to the paper titled: 
"BERT-based Idiom Identification using Language Translation and Word Cohesion".

## src Directory

### Jupyter Notebooks:

1. `align_err_analysis.ipynb`: Utilized to align incorrect predictions of the model on the test set for error analysis.
2. `scrape_theidioms.ipynb`: Used to scrape, preprocess, and annotate the 'theidioms' dataset from https://www.theidioms.com/.

### Python Scripts:

1. `script.py`: Example implementation of our model on the 'vnc' dataset, utilizing combined loss as the custom loss function.
2. `generalized.py`: Experiment on generalization, trained on the 'theidioms' dataset and tested on the 'gtrans' dataset.

## Error Analysis

This folder contains subfolders containing the files for the `formal`, `magpie`, and `VNC` datasets for which Error Analysis and Semantic Accuracy computation is performed.

## Datasets

The datasets can be acquired as follows :

*Note : some additional preprocessing will be required to exactly match our task*

1. magpie : https://github.com/hslh/magpie-corpus
2. VNC : http://www.natcorp.ox.ac.uk/
3. theidioms : Extracted and preprocessed using the `scrape.ipynb` notebook.
4. formal : https://github.com/prateeksaxena2809/EPIE_Corpus
