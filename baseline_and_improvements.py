'''
Topic - FEAT: From Frequency-based Emotion Analysis to Transformers

Description: The code below demonstrates the process of fine-tuning a pre-trained transformer 
model (DistilRoBERTa) for emotion classification using the Hugging Face Transformers library. 
We have included functions for preprocessing the dataset, tokenizing text data, creating data 
loaders, training and evaluating both neural network and machine learning models, and extracting 
features for machine learning models. We also evaluated the performance of different machine 
learning algorithms such as Support Vector Classifier (SVC), Random Forest Classifier, Gradient 
Boosting Classifier, and AdaBoost Classifier on both raw and fused features extracted from the 
pre-trained transformer model and TF-IDF representations. Additionally, we compared the 
performance of the baseline transformer model with an improved version that incorporates 
additional layers for feature extraction. Overall, the code provides a comprehensive framework 
for experimenting with both neural network and combination of transformers with machine learning 
models approaches for emotion classification tasks.
'''

# import necessary libraries
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel
import numpy as np
from datasets import Dataset
from datasets.dataset_dict import DatasetDict
import os
import evaluate
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_scheduler
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.feature_extraction.text import TfidfVectorizer

def preprocess_function(examples):
    '''
    This function tokenizes the text column in the examples dictionary and returns the tokenized text.
    It also pads the tokenized text to the maximum length and truncates the text if it exceeds the maximum length.
    '''
    return tokenizer(examples["text"], padding="max_length", truncation=True)
 
def get_tokenized_train_test_set(train, test):
    '''
    This function tokenizes the text column in the train and test datasets by using map function which applies the preprocess_function to the datasets.
    It also removes the text column from the datasets and renames the label column to labels.
    '''
    tokenized_train = train.map(preprocess_function, batched=True).remove_columns(["text"]).rename_column("label", "labels") # tokenizing the train dataset
    tokenized_test = test.map(preprocess_function, batched=True).remove_columns(["text"]).rename_column("label", "labels") # tokenizing the test dataset
    return tokenized_train, tokenized_test # returning the tokenized train and test datasets

def get_data_loader(tokenized_train, tokenized_test, batch_size=16):
    '''
    This function creates the train and test dataloaders by using the DataLoader class from torch.utils.data.
    It shuffles the train dataloader and uses the batch size to create the dataloader.
    No shuffling is done for the test dataloader.
    '''
    train_dataloader = DataLoader(tokenized_train, shuffle=True, batch_size=batch_size) # creating the train dataloader
    test_dataloader = DataLoader(tokenized_test,  batch_size=batch_size) # creating the test dataloader
    return train_dataloader, test_dataloader # returning the train and test dataloaders

def train_and_evaluate_model(model, train_loader, test_loader, learning_rate, epochs, weight_decay, return_model=False):
    '''
    This function trains the model on the train_loader and evaluates the model on the test_loader.
    '''
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay) # using AdamW optimizer with learning rate and weight decay
    num_train_steps = epochs * len(train_loader) # number of training steps is equal to the number of epochs times the length of the train loader
    lr_scheduler = get_scheduler(
        name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_train_steps
    ) # using linear scheduler with no warmup steps and the number of training steps
    loss_fn = nn.CrossEntropyLoss() # using CrossEntropyLoss as the loss function

    progress_bar = tqdm(range(num_train_steps)) # using tqdm to display the progress bar

    model.train() # setting the model to training mode
    for _ in range(epochs): # iterating over the number of epochs
        for batch in train_loader: # iterating over the train loader
            batch = {k: v.to(device) for k, v in batch.items()} # moving the batch to the device
            logits = model(**batch) # getting the logits from the model
            # NLP Concept(||): Probabilistic Model
            loss = loss_fn(logits.view(-1, 6), batch["labels"].view(-1)) # calculating the loss
            loss.backward() # backpropagating the loss

            optimizer.step() # taking a step with the optimizer
            lr_scheduler.step() # taking a step with the learning rate scheduler
            optimizer.zero_grad() # zeroing the gradients
            progress_bar.update(1) # updating the progress bar

    acc = evaluate.load('accuracy') # loading the accuracy metric
    f1 = evaluate.load('f1') # loading the f1 metric

    model.eval() # setting the model to evaluation mode
    with torch.no_grad(): # turning off gradient calculations
        for batch in test_loader: # iterating over the test loader
            batch = {k: v.to(device) for k, v in batch.items()} # moving the batch to the device
            logits = model(**batch) # getting the logits from the model
            predictions = torch.argmax(logits, dim=-1) # getting the predictions
            acc.add_batch(predictions=predictions, references=batch["labels"]) # adding the batch to the accuracy metric
            f1.add_batch(predictions=predictions, references=batch["labels"]) # adding the batch to the f1 metric

    if return_model: # if return_model is True, return the model along with the metrics
        return model, acc.compute(), f1.compute(average='macro') # return the model, accuracy, and f1 score
    
    return acc.compute(), f1.compute(average='macro') # return the accuracy and f1 score

# defining the EmotionClassifier class
class EmotionClassifier(nn.Module):
    def __init__(self, model_name, num_labels):
        super(EmotionClassifier, self).__init__()
        self.model = AutoModel.from_pretrained(model_name) # loading the pre-trained model
        self.classifier = nn.Linear(self.model.config.hidden_size, num_labels) # adding a linear layer with the number of labels

    def forward(self, input_ids, attention_mask, labels): 
        outputs = self.model(input_ids, attention_mask=attention_mask) # getting the outputs from the model
        last_hidden_state = outputs[0] # getting the last hidden state
        cls_token = last_hidden_state[:, 0, :] # getting the CLS token
        logits = self.classifier(cls_token) # getting the logits
        return logits # returning the logits
    
# defining the EmotionClassifierImproved class
class EmotionClassifierImproved(nn.Module):
    def __init__(self, model_name, num_labels):
        super(EmotionClassifierImproved, self).__init__()
        self.model = AutoModel.from_pretrained(model_name) # loading the pre-trained model
        # NLP Concept(|): Classification
        self.classifier = nn.Linear(self.model.config.hidden_size, num_labels) # adding a linear layer with the number of labels
        self.dropout = nn.Dropout(0.1) # adding a dropout layer
        self.hidden = nn.Linear(self.model.config.hidden_size, self.model.config.hidden_size) # adding a hidden layer

    def forward(self, input_ids, attention_mask, labels): 
        outputs = self.model(input_ids, attention_mask=attention_mask) # getting the outputs from the model
        last_hidden_state = outputs[0] # getting the last hidden state
        x = last_hidden_state[:, 0, :] # getting the CLS token
        x = self.dropout(x) # applying dropout
        x = self.hidden(x) # applying the hidden layer
        x = torch.tanh(x) # applying the tanh activation function
        x = self.dropout(x) # applying dropout
        x = self.hidden(x) # applying the hidden layer
        x = torch.tanh(x) # applying the tanh activation function
        logits = self.classifier(x) # getting the logits
        return logits  # returning the logits
        
def extract_feats_for_ml(model, train_loader, test_loader):
    '''
    This function extracts the features from the model for the train and test loaders.
    '''
    train = {'embeds': [], 'labels': []} # creating a dictionary to store the embeddings and labels for the train set
    test = {'embeds': [], 'labels': []} # creating a dictionary to store the embeddings and labels for the test set
    
    progress_bar = tqdm(range(len(train_loader))) # using tqdm to display the progress bar

    model.eval() # setting the model to evaluation mode
    with torch.no_grad(): # turning off gradient calculations
        for batch in train_loader: # iterating over the train loader
            batch = {k: v.to(device) for k, v in batch.items()} # moving the batch to the device
            embeds = model(batch['input_ids'], batch['attention_mask'])[0][:, 0, :] # getting the embeddings
            train['embeds'].append(embeds.cpu()) # appending the embeddings to the train dictionary
            train['labels'].append(batch['labels'].cpu()) # appending the labels to the train dictionary
            progress_bar.update(1) # updating the progress bar

    progress_bar = tqdm(range(len(test_loader))) # using tqdm to display the progress bar
    
    model.eval() # setting the model to evaluation mode
    with torch.no_grad(): # turning off gradient calculations
        for batch in test_loader: # iterating over the test loader
            batch = {k: v.to(device) for k, v in batch.items()} # moving the batch to the device
            embeds = model(batch['input_ids'], batch['attention_mask'])[0][:, 0, :] # getting the embeddings
            test['embeds'].append(embeds.cpu()) # appending the embeddings to the test dictionary
            test['labels'].append(batch['labels'].cpu()) # appending the labels to the test dictionary
            progress_bar.update(1) # updating the progress bar

    train['embeds'] = torch.cat(train['embeds'], dim=0) # concatenating the embeddings for the train set
    train['labels'] = torch.cat(train['labels'], dim=0) # concatenating the labels for the train set
    test['embeds'] = torch.cat(test['embeds'], dim=0) # concatenating the embeddings for the test set
    test['labels'] = torch.cat(test['labels'], dim=0) # concatenating the labels for the test set
    
    train['embeds'] = train['embeds'].numpy() # converting the embeddings to numpy arrays
    train['labels'] = train['labels'].numpy() # converting the labels to numpy arrays

    test['embeds'] = test['embeds'].numpy() # converting the embeddings to numpy arrays
    test['labels'] = test['labels'].numpy() # converting the labels to numpy arrays

    return train, test # returning the train and test dictionaries

def get_tfidf_features(train, test):
    '''
    This function extracts the TF-IDF features from the text column in the train and test datasets.
    '''
    vectorizer = TfidfVectorizer() # using the TfidfVectorizer to extract the TF-IDF features
    train_feats = vectorizer.fit_transform(train['text']) # fitting and transforming the train text
    test_feats = vectorizer.transform(test['text']) # transforming the test text
    return train_feats.toarray(), test_feats.toarray() # returning the train and test TF-IDF features

def fuse_feats(feats1, feats2):
    '''
    This function fuses the features by concatenating them along the columns.
    '''
    fused_feats = np.concatenate((feats1, feats2), axis=1) # concatenating the features along the columns
    return fused_feats # returning the fused features

def train_and_evaluate_ml_model(model, train, test):
    '''
    This function trains the model on the train set and evaluates it on the test set.
    '''
    model.fit(train['embeds'], train['labels']) # fitting the model on the train set
    preds = model.predict(test['embeds']) # getting the predictions on the test set
    acc = accuracy_score(test['labels'], preds) # calculating the accuracy
    f1 = f1_score(test['labels'], preds, average='macro') # calculating the f1 score
    return acc, f1 # returning the accuracy and f1 score

def train_and_test_ml_models(train, test, fused_feats=False, fused_feats_train=None, fused_feats_test=None):
    '''
    This function trains and evaluates the ML models on the train and test sets.
    '''
    models = [SVC(), RandomForestClassifier(), HistGradientBoostingClassifier(), AdaBoostClassifier()] # defining the ML models

    for model in models: # iterating over the models
        if fused_feats: # if fused_feats is True, train and evaluate the model using the fused features
            acc, f1 = train_and_evaluate_ml_model(model, {'embeds': fused_feats_train, 'labels': train['labels']}, {'embeds': fused_feats_test, 'labels': test['labels']}) 
            print(f"ML Model Performance ({model.__class__.__name__} with Fused Feats) - Accuracy {acc}, F1: {f1}")
        else: # otherwise, train and evaluate the model using the extracted features
            acc, f1 = train_and_evaluate_ml_model(model, train, test)
            print(f"ML Model Performance ({model.__class__.__name__}) - Accuracy {acc}, F1: {f1}")


# NLP Concept(III): Transformers
MODEL_NAME = "distilbert/distilroberta-base" # defining the model name

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu") # setting the device to cuda if it is available, otherwise to cpu

# defining the hyperparameters
batch_size = 16
epochs = 2
learning_rate = 2e-5
weight_decay = 0.01

# NLP Concept(IV): Applications (Sentiment Analysis)
dataset = load_dataset("dair-ai/emotion", name="unsplit") # loading the dataset

# splitting the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(dataset['train']['text'], dataset['train']['label'], test_size=0.2, random_state=42, stratify=dataset['train']['label'])

train_data = {"text": X_train, "label": y_train} # creating the train data dictionary
test_data = {"text": X_test, "label": y_test} # creating the test data dictionary

train_dataset = Dataset.from_dict(train_data) # creating the Dataset object from the train data dictionary for the train set
test_dataset = Dataset.from_dict(test_data) # creating the Dataset object from the test data dictionary for the test set

dataset = DatasetDict({"train": train_dataset, "test": test_dataset}) # creating the DatasetDict object from the train and test datasets

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME) # loading the tokenizer

tokenized_train, tokenized_test = get_tokenized_train_test_set(dataset["train"], dataset["test"]) # getting the tokenized train and test datasets

tokenized_train.set_format('torch') # setting the format of the tokenized train dataset to torch
tokenized_test.set_format('torch') # setting the format of the tokenized test dataset to torch

train_loader, test_loader = get_data_loader(tokenized_train, tokenized_test, batch_size=batch_size) # getting the train and test dataloaders

baseline = EmotionClassifier(MODEL_NAME, num_labels=6).to(device) # creating the baseline model and moving it to the device
model, acc, f1 = train_and_evaluate_model(baseline, train_loader, test_loader, learning_rate, epochs, weight_decay, return_model=True) # training and evaluating the baseline model

print(f"Baseline Performance (EmotionClassifier) - Accuracy {acc['accuracy']}, F1: {f1['f1']}")

train, test = extract_feats_for_ml(model.model, train_loader, test_loader) # extracting the features for the ML models from the fine-tuned baseline model

print(f'================ Results for ML models trained on features extracted from the Baseline model ================')

train_and_test_ml_models(train, test) # training and testing the ML models on the extracted features

tfidf_train, tfidf_test = get_tfidf_features(dataset['train'], dataset['test']) # extracting the TF-IDF features from the text column in the train and test datasets

fused_feats_train = fuse_feats(train['embeds'], tfidf_train) # fusing the features from the fine-tuned baseline model and the TF-IDF features for the train set
fused_feats_test = fuse_feats(test['embeds'], tfidf_test) # fusing the features from the fine-tuned baseline model and the TF-IDF features for the test set

print(f'================ Results for ML models trained on fused features extracted from the Baseline model ================')

train_and_test_ml_models(train, test, fused_feats=True, fused_feats_train=fused_feats_train, fused_feats_test=fused_feats_test) # training and testing the ML models on the fused features

improved = EmotionClassifierImproved(MODEL_NAME, num_labels=6).to(device) # creating the improved model and moving it to the device
model, acc, f1 = train_and_evaluate_model(improved, train_loader, test_loader, learning_rate, epochs, weight_decay, return_model=True) # training and evaluating the improved model

print(f"Improved Performance (EmotionClassifierImproved) - Accuracy {acc['accuracy']}, F1: {f1['f1']}")

train, test = extract_feats_for_ml(model.model, train_loader, test_loader) # extracting the features for the ML models from the fine-tuned improved model

print(f'================ Results for ML models trained on features extracted from the Improved model ================')

train_and_test_ml_models(train, test) # training and testing the ML models on the extracted features

fused_feats_train = fuse_feats(train['embeds'], tfidf_train) # fusing the features from the fine-tuned improved model and the TF-IDF features for the train set
fused_feats_test = fuse_feats(test['embeds'], tfidf_test) # fusing the features from the fine-tuned improved model and the TF-IDF features for the test set

print(f'================ Results for ML models trained on fused features extracted from the Improved model ================')

train_and_test_ml_models(train, test, fused_feats=True, fused_feats_train=fused_feats_train, fused_feats_test=fused_feats_test) # training and testing the ML models on the fused features