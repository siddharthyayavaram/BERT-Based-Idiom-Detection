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

```
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
4. `unfiltered_dict.json` : An unfiltered dictionary stored as a json file, with the idiom as the key corresponding to a list of sentences.

*For access to the VNC dataset files, please contact f20213116@hyderabad.bits-pilani.ac.in via email*

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


## Citation

If you find this work helpful for your research please consider citing the following bibtex entry.

```
@inproceedings{yayavaram-etal-2024-bert,
    title = "{BERT}-based Idiom Identification using Language Translation and Word Cohesion",
    author = "Yayavaram, Arnav  and
      Yayavaram, Siddharth  and
      Upadhyay, Prajna Devi  and
      Das, Apurba",
    editor = {Bhatia, Archna  and
      Bouma, Gosse  and
      Do{\u{g}}ru{\"o}z, A. Seza  and
      Evang, Kilian  and
      Garcia, Marcos  and
      Giouli, Voula  and
      Han, Lifeng  and
      Nivre, Joakim  and
      Rademaker, Alexandre},
    booktitle = "Proceedings of the Joint Workshop on Multiword Expressions and Universal Dependencies (MWE-UD) @ LREC-COLING 2024",
    month = may,
    year = "2024",
    address = "Torino, Italia",
    publisher = "ELRA and ICCL",
    url = "https://aclanthology.org/2024.mwe-1.26",
    pages = "220--230",
    abstract = "An idiom refers to a special type of multi-word expression whose meaning is figurative and cannot be deduced from the literal interpretation of its components. Idioms are prevalent in almost all languages and text genres, necessitating explicit handling by comprehensive NLP systems. Such phrases are referred to as Potentially Idiomatic Expressions (PIEs) and automatically identifying them in text is a challenging task. In this paper, we propose using a BERT-based model fine-tuned with custom objectives, to improve the accuracy of detecting PIEs in text. Our custom loss functions capture two important properties (word cohesion and language translation) to distinguish PIEs from non-PIEs. We conducted several experiments on 7 datasets and showed that incorporating custom objectives while training the model leads to substantial gains. Our models trained using this approach also have better sequence accuracy over DISC, a state-of-the-art PIE detection technique, along with good transfer capabilities.",
}
```
