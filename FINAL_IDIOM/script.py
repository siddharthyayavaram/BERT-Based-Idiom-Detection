from datasets import DatasetDict, Dataset

import datasets
import numpy as np
from transformers import BertTokenizerFast
from transformers import DataCollatorForTokenClassification
from transformers import AutoModelForTokenClassification
from sklearn.metrics import classification_report

text_data = []
tag_data = []

with open('Formal_Idioms_Words.txt', 'r', encoding='utf-8') as file:
    # Read lines and remove newline characters
    text_data = [line.rstrip('\n').split() for line in file.readlines()]

with open('Formal_Idioms_Tags.txt', 'r', encoding='utf-8') as file:
    # Read lines and remove newline characters
    tag_data = [line.rstrip('\n').split() for line in file.readlines()]

def convert_tags_to_integers_list(list_of_tag_lists):
    tag_mapping = {'O': 0, 'B-IDIOM': 1,'I-IDIOM':1}
    return [[tag_mapping[tag] for tag in inner_list] for inner_list in list_of_tag_lists]

def convert_tags_to_integers_list_1(list_of_tag_lists):
    tag_mapping = {'0': 0, '1': 1}
    return [[tag_mapping[tag] for tag in inner_list] for inner_list in list_of_tag_lists]

tag_data = convert_tags_to_integers_list(tag_data)

import random

# Calculate the split sizes
total_samples = len(text_data)
train_size = int(0.8 * total_samples)
val_size = int(0.1 * total_samples)
test_size = total_samples - train_size - val_size
random.seed(42)
# Shuffle the data
combined_data = list(zip(text_data, tag_data))
random.shuffle(combined_data)

# Separate the data into training, validation, and test sets
train_data = combined_data[:train_size]
val_data = combined_data[train_size:train_size + val_size]
x = train_size + val_size
test_data = combined_data[x: x + test_size]

# Convert back to separate lists for tokens and ner_tags
train_tokens, train_ner_tags = zip(*train_data)
val_tokens, val_ner_tags = zip(*val_data)
test_tokens, test_ner_tags = zip(*test_data)

# Create the final dictionaries with ids starting from 0
train = {"id": list(map(str, range(train_size))), "tokens": list(train_tokens), "ner_tags": list(train_ner_tags)}
validation = {"id": list(map(str, range(val_size))), "tokens": list(val_tokens), "ner_tags": list(val_ner_tags)}
test = {"id": list(map(str, range(test_size))), "tokens": list(test_tokens), "ner_tags": list(test_ner_tags)}

# Create a DatasetDict
dataset_dict = DatasetDict({
    "train": Dataset.from_dict(train),
    "validation": Dataset.from_dict(validation),
    "test": Dataset.from_dict(test),
})

tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

def tokenize_and_align_labels(examples, label_all_tokens=True):
    tokenized_inputs = tokenizer(examples["tokens"], truncation=True, is_split_into_words=True)
    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)

        previous_word_idx = None
        label_ids = []

        for word_idx in word_ids:
            if word_idx is None:
                # set –100 as the label for these special tokens
                label_ids.append(-100)
            # For the other tokens in a word, we set the label to either the current label or -100, depending on
            # the label_all_tokens flag.
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            else:
                label_ids.append(label[word_idx] if label_all_tokens else -100)
                # mask the subword representations after the first subword

            previous_word_idx = word_idx
        labels.append(label_ids)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

import torch

torch.manual_seed(42)

tokenized_datasets = dataset_dict.map(tokenize_and_align_labels, batched=True)

model = AutoModelForTokenClassification.from_pretrained("bert-base-uncased", num_labels=2)

# Check for GPU availability
device = "cuda" if torch.cuda.is_available() else "cpu"

# Move the model to the GPU
model = model.to(device)

from transformers import TrainingArguments, Trainer

data_collator = DataCollatorForTokenClassification(tokenizer)

metric = datasets.load_metric("seqeval")

example = dataset_dict['train'][0]
label_list = ['O','IDIOM']

labels = [label_list[i] for i in example["ner_tags"]]

metric.compute(predictions=[labels], references=[labels])

def compute_metrics(eval_preds):
    """
    Function to compute the evaluation metrics for Named Entity Recognition (NER) tasks.
    The function computes precision, recall, F1 score and accuracy.

    Parameters:
    eval_preds (tuple): A tuple containing the predicted logits and the true labels.

    Returns:
    A dictionary containing the precision, recall, F1 score and accuracy.
    """
    pred_logits, labels = eval_preds

    pred_logits = np.argmax(pred_logits, axis=2)
    # the logits and the probabilities are in the same order,
    # so we don’t need to apply the softmax

    # We remove all the values where the label is -100
    predictions = [
        [label_list[eval_preds] for (eval_preds, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(pred_logits, labels)
    ]

    true_labels = [
      [label_list[l] for (eval_preds, l) in zip(prediction, label) if l != -100]
       for prediction, label in zip(pred_logits, labels)
   ]

    flat_predictions = [label for sublist in predictions for label in sublist]
    flat_true_labels = [label for sublist in true_labels for label in sublist]

    report = classification_report(flat_true_labels, flat_predictions,digits = 4)

    print(report)

    results = metric.compute(predictions=predictions, references=true_labels)
    return {
   "precision": results["overall_precision"],
   "recall": results["overall_recall"],
   "f1": results["overall_f1"],
  "accuracy": results["overall_accuracy"],
    }

import matplotlib.pyplot as plt

losses_list = []

import torch
from transformers import BertTokenizer, BertModel
import numpy as np

from sklearn.metrics.pairwise import cosine_similarity

import spacy

# Load the spaCy English model
nlp = spacy.load("en_core_web_md")

def get_bert_embeddings(text, model, tokenizer):
    input_ids = tokenizer.encode(text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(input_ids)
    return outputs.last_hidden_state[0].mean(dim=0).cpu().numpy()

def compute_semantic_relatedness(word1_embedding, word2_embedding):
    return cosine_similarity([word1_embedding], [word2_embedding])[0, 0]

def filter_pos(words):
    # Keep only nouns and verbs
    pos_tags_to_keep = ["NOUN", "VERB"]
    filtered_words = []
    for word in words:
        doc = nlp(word)
        for token in doc:
            if token.pos_ in pos_tags_to_keep:
                filtered_words.append(word)
                break
    return filtered_words

def calculate_cohesion_score(context_words, idiom_words, model, tokenizer):
    # Filter out only nouns and verbs from context words
    filtered_context_words = filter_pos(context_words)

    word_embeddings = {}

    # Get BERT embeddings for all filtered words in the context
    for word in filtered_context_words:
        word_embeddings[word] = get_bert_embeddings(word, model, tokenizer)

    cohesion_graph = np.zeros((len(filtered_context_words), len(filtered_context_words)))

    # Populate cohesion graph with semantic relatedness scores
    for i, word1 in enumerate(filtered_context_words):
        for j, word2 in enumerate(filtered_context_words):
            cohesion_graph[i, j] = compute_semantic_relatedness(word_embeddings[word1], word_embeddings[word2])

    # Calculate connectivity
    connectivity = np.mean(cohesion_graph)

    idiom_indices = [filtered_context_words.index(word) for word in idiom_words if word in filtered_context_words]

    # print(idiom_indices)

    if idiom_indices:
        cohesion_graph = np.delete(cohesion_graph, idiom_indices, axis=0)
        cohesion_graph = np.delete(cohesion_graph, idiom_indices, axis=1)

        # print(cohesion_graph)

        # Compare connectivity changes
        connectivity_without_idiom = np.mean(cohesion_graph)

        # print(str(connectivity_without_idiom)+"2")

        if connectivity_without_idiom > connectivity:
            return 'idiom', connectivity, connectivity_without_idiom
        else:
            return 'literal', connectivity, connectivity_without_idiom
    else:
        # # Handle the case where no idiom words are found in the filtered context
        # return 'cant embed so idiom'

            filtered_context_words = context_words

            # print(filtered_context_words)

            word_embeddings = {}

            # Get BERT embeddings for all filtered words in the context
            for word in filtered_context_words:
                word_embeddings[word] = get_bert_embeddings(word, model, tokenizer)

            cohesion_graph = np.zeros((len(filtered_context_words), len(filtered_context_words)))

            # Populate cohesion graph with semantic relatedness scores
            for i, word1 in enumerate(filtered_context_words):
                for j, word2 in enumerate(filtered_context_words):
                    cohesion_graph[i, j] = compute_semantic_relatedness(word_embeddings[word1], word_embeddings[word2])

            # Calculate connectivity
            connectivity = np.mean(cohesion_graph)

            idiom_indices = [filtered_context_words.index(word) for word in idiom_words if word in filtered_context_words]

            # print(idiom_indices)

            if idiom_indices:
                cohesion_graph = np.delete(cohesion_graph, idiom_indices, axis=0)
                cohesion_graph = np.delete(cohesion_graph, idiom_indices, axis=1)

                # print(cohesion_graph)

                # Compare connectivity changes
                connectivity_without_idiom = np.mean(cohesion_graph)

                # print(str(connectivity_without_idiom)+"4")

                if connectivity_without_idiom > connectivity:
                    return 'idiom', connectivity, connectivity_without_idiom
                else:
                    return 'literal', connectivity, connectivity_without_idiom
            else:
                return 'idiom',0,0


model_name_1 = 'bert-base-uncased'
tokenizer_1 = BertTokenizer.from_pretrained(model_name_1)
model_1 = BertModel.from_pretrained(model_name_1)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_1 = model_1.to(device)

def idiom_part(ip_ids,labels):
    idiom_list = []
    for index, value in enumerate(labels.view(-1)):
        if value == 1:
            idiom_list.append(ip_ids.view(-1)[index])

    return tokenizer.decode(ip_ids.view(-1)).replace('[CLS]','').replace('[SEP]','').split(),tokenizer.decode(idiom_list).split()

from googletrans import Translator

translator=Translator()

from nltk.translate import meteor_score

import nltk
nltk.download('wordnet')

def meteor(example):
    translator = Translator()
    sen = example
    translated_text = translator.translate(sen, dest='hi', timeout=10).text
    back_translated_text = translator.translate(translated_text, dest='en', timeout=10).text
    bsen = back_translated_text
    r = [sen.split()]
    c = bsen.split()
    meteor_score_value = meteor_score.meteor_score(r, c)

    return meteor_score_value, sen, bsen

class MyTrainer_Trans(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):

        # print("------------------------------------------------------------")

        ip_ids = inputs['input_ids']
        labels = inputs.pop("labels")

        b = 0
        a = 0
        c = 0

        m,n = idiom_part(ip_ids,labels)

        if(len(n)!=0):
          c,a,b = calculate_cohesion_score(m,n,model_1,tokenizer_1)

        x = ' '.join(m)

        y,i,j = meteor(x)

        outputs = model(**inputs)
        logits = outputs.logits

        # Calculate the cross-entropy loss
        loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.shape[-1]), labels.view(-1), ignore_index=-100)

        if y < 0.7:
           loss = loss*100

        if b-a > 0.02:
            loss = loss*100

        losses_list.append(loss)
        if return_outputs:
            return loss, outputs
        else:
            return loss


# Example usage
args = TrainingArguments(
    "test-ner",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    num_train_epochs=3,
    weight_decay=0.01,
    seed=42
)

# Create an instance of your custom trainer
ner_trainer_trans = MyTrainer_Trans(
    model=model,
    args=args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

np.seterr(all='ignore')

import warnings

warnings.filterwarnings("ignore", category = RuntimeWarning)

ner_trainer_trans.train()

label_list = ['O', 'IDIOM']

# Predict on the test dataset
test_predictions = ner_trainer_trans.predict(tokenized_datasets["test"])

# Extract predicted labels from test_predictions
predicted_labels = test_predictions.predictions

pred_logits = np.argmax(test_predictions.predictions, axis=2)

from pprint import pprint

labels = tokenized_datasets["test"]['labels']

predictions = [
    [label_list[eval_preds] for (eval_preds, l) in zip(prediction, label) if l != -100]
    for prediction, label in zip(pred_logits, labels)
]

true_labels = [
  [label_list[l] for (eval_preds, l) in zip(prediction, label) if l != -100]
    for prediction, label in zip(pred_logits, labels)
]

flat_predictions = [label for sublist in predictions for label in sublist]
flat_true_labels = [label for sublist in true_labels for label in sublist]

report = classification_report(flat_true_labels, flat_predictions,digits = 4, output_dict = True)

dict = {}
for i,(true_sublabels,pred_sublabels) in enumerate(zip(true_labels, predictions)):
    dict[i] = classification_report(true_sublabels, pred_sublabels, digits=4, output_dict = True, zero_division = 1)

not_1 = {}
ctr=0

for key, inner_dict in dict.items():
    if inner_dict['weighted avg']['f1-score'] != 1.0:
        not_1[key] = inner_dict
    else:
        ctr+=1

print("CTR === ",ctr)
print(len(not_1))

indices = []
for i,d in not_1.items():
    indices.append(i)

print(indices)
print(len(indices))

# Open a text file for writing
with open("error_analysis_formal_combined_loss.txt", "w") as file:
    # Iterate over indices
    for x in indices:
        prediction = predictions[x]
        tokens = tokenized_datasets["test"][x]['tokens']
        labels = tokenized_datasets["test"][x]['labels'][1:-1]  # Ignore the first and last elements in labels

        # Align tokens and labels
        max_length = max(len(tokens), len(labels))
        tokens += [''] * (max_length - len(tokens))
        labels += [''] * (max_length - len(labels))

        # Write the aligned output to the file
        for token, label, pred in zip(tokens, labels, prediction):
            file.write(f"Prediction: {pred.ljust(10)} Token: {token.ljust(15)} Label: {label}\n")
        file.write("\n")  # Add a newline to separate each entry