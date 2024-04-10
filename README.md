# BERT-based Idiom Identification using Language Translation and Word Cohesion

This repository is intended to provide a collection of the files related to the paper titled: 

"BERT-based Idiom Identification using Language Translation and Word Cohesion".

### About the Project

In this paper, we propose using a BERT- based model fine-tuned with the custom objectives to improve the accuracy of detecting PIEs in text. Our custom loss functions capture two important properties (word cohesion and language translation) to distinguish PIEs from non-PIEs. We conducted several experiments on 7 datasets and showed that incorporating custom objectives while training the model leads to substantial gains.

This paper has been accepted for publication at MWE-UD 2024, to be held in conjunction with LREC-COLING.

### Built With

Our code relies on the following resources/libraries:

- PyTorch
- Huggingface's Transformer
- NLTK
- spacy

### Setup

Python version used is Python 3.9.12

Run this command to install all required libraries

```bash
pip install -r requirements.txt

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

1. `theidioms_sentences.txt`: A collection of 7830 sentences with idiomatic expressions. Each idiomatic expression appears in an average of 4.87 sentences. There are 1606 unique idioms present.
2. `theidioms1_1_sentences.txt`: A collection of 1606 sentences where each sentence has a unique idiom.
3. `list_of_idioms.txt`: A list of all 1606 idiomatic expressions which appear in the above datasets.

The remaining datasets are all publicly available and can be acquired as follows :

*Note : some additional preprocessing will be required to exactly match our task*

1. magpie : https://github.com/hslh/magpie-corpus
2. VNC : http://www.natcorp.ox.ac.uk/
3. theidioms : Extracted and preprocessed using the `scrape_theidioms.ipynb` notebook.
4. formal : https://github.com/prateeksaxena2809/EPIE_Corpus


## Compute Resources Utilized

For training and testing our models, we make use of the following compute resources:

- **Specs:**
  - Processor: 32 × 2 cores AMD EPYC 50375
  - RAM: 1 TB
  - GPUs: 8x NVIDIA A100 SXM4 80GB
