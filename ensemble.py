'''
Topic - FEAT: From Frequency-based Emotion Analysis to Transformers

Description: This code is a machine learning pipeline for emotion classification. It loads a dataset, 
tokenizes it using two different transformer models (distilroBERTa and distilBERT), then trains an ensemble model 
using these tokenized inputs. The ensemble model combines the outputs of the two models and trains on emotion labels. 
Features are extracted from the ensemble model for further use in traditional machine learning models like SVM, Random Forest, 
etc. Finally, these ML models are trained and evaluated on the extracted features. The pipeline includes training, evaluation, 
and comparison of both deep learning and traditional ML models for emotion classification.
'''

import torch
import numpy as np
import os
import math
import warnings
from tqdm.auto import tqdm
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
from datasets import load_dataset, load_metric, Dataset, DatasetDict
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, DataCollatorWithPadding, AutoModel, get_scheduler
import evaluate
import torch.nn as nn

warnings.filterwarnings('ignore')
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Load dataset
#NLP Concept: Application (Sentiment Analysis)
dataset = load_dataset("dair-ai/emotion", name="unsplit")

def get_tokenized_train_test_set(train, test, tokenizer):
    '''
    Tokenizes the training and test sets using the provided tokenizer.

    Args:
    - train (Dataset): Training dataset.
    - test (Dataset): Test dataset.
    - tokenizer (AutoTokenizer): Tokenizer to be used.

    Returns:
    - tokenized_train (Dataset): Tokenized training dataset.
    - tokenized_test (Dataset): Tokenized test dataset.
    '''
    def preprocess_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)
    tokenized_train = train.map(preprocess_function, batched=True).remove_columns(["text"]).rename_column("label", "labels")
    tokenized_test = test.map(preprocess_function, batched=True).remove_columns(["text"]).rename_column("label", "labels")
    return tokenized_train, tokenized_test

def train_and_evaluate_model(model, train_loader1, test_loader1, train_loader2, test_loader2, learning_rate, epochs, weight_decay):
    '''
    Trains and evaluates the ensemble model.

    Args:
    - model (EmotionEnsembleClassifier): Ensemble model to be trained.
    - train_loader1 (DataLoader): DataLoader for the first transformer model.
    - test_loader1 (DataLoader): DataLoader for the first transformer model.
    - train_loader2 (DataLoader): DataLoader for the second transformer model.
    - test_loader2 (DataLoader): DataLoader for the second transformer model.
    - learning_rate (float): Learning rate for optimization.
    - epochs (int): Number of training epochs.
    - weight_decay (float): Weight decay for optimization.

    Returns:
    - accuracy (float): Accuracy of the trained model.
    - macro_f1 (float): Macro F1 score of the trained model.
    - embeds_train (list): List of embeddings for training set.
    - embeds_test (list): List of embeddings for test set.
    '''
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    num_train_steps = epochs * len(train_loader1)
    lr_scheduler = get_scheduler(
        name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_train_steps
    )
    loss_fn = nn.CrossEntropyLoss()

    progress_bar = tqdm(range(num_train_steps))
    embeds_train = []
    model.train()
    for epoch in range(epochs):
        for step, combined_batch in enumerate(zip(train_loader1, train_loader2)):
            batch1,batch2 = combined_batch
            batch_1 = tuple(t.to(device) for k,t in batch1.items())
            batch_2 = tuple(t.to(device) for k,t in batch2.items())
            inputs = {
                "input_ids": [batch_1[1], batch_2[1]],
                "attention_mask": [batch_1[2], batch_2[2]],
                "labels": batch_1[0]
            }
            logits = model(**inputs)
            if epoch == 0:
                embeds_train.extend(model.last_layer)
            loss = loss_fn(logits.view(-1, 6), inputs["labels"].view(-1))
            loss.backward()

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)

    acc = evaluate.load('accuracy')
    f1 = evaluate.load('f1')
    embeds_test = []
    model.eval()
    with torch.no_grad():
        for step, combined_batch in enumerate(zip(test_loader1, test_loader2)):
            batch1,batch2 = combined_batch
            batch_1 = tuple(t.to(device) for k,t in batch1.items())
            batch_2 = tuple(t.to(device) for k,t in batch2.items())
            inputs = {
                "input_ids": [batch_1[1], batch_2[1]],
                "attention_mask": [batch_1[2], batch_2[2]],
                "labels": batch_1[0]
            }
            logits = model(**inputs)
            embeds_test.extend(model.last_layer)
            predictions = torch.argmax(logits, dim=-1)
            acc.add_batch(predictions=predictions, references=inputs["labels"])
            f1.add_batch(predictions=predictions, references=inputs["labels"])

    return acc.compute(), f1.compute(average='macro'), embeds_train, embeds_test

class EmotionEnsembleClassifier(nn.Module):
    '''
    Emotion Ensemble Classifier combining two transformer models.
    '''
    def __init__(self, model_name1, model_name2, num_labels):
        super(EmotionEnsembleClassifier, self).__init__()
        self.model1 = AutoModel.from_pretrained(model_name1)
        self.model2 = AutoModel.from_pretrained(model_name2)
        self.last_layer = []

        #NLP Concept: Classification
        self.classifier = nn.Linear(self.model1.config.hidden_size+self.model2.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask, labels):
        outputs1 = self.model1(input_ids[0], attention_mask=attention_mask[0])
        outputs2 = self.model2(input_ids[1], attention_mask=attention_mask[1])
        last_hidden_state1 = outputs1[0]
        last_hidden_state2 = outputs2[0]
        cls_token1 = last_hidden_state1[:, 0, :]
        cls_token2 = last_hidden_state2[:, 0, :]
        tr = torch.cat((cls_token1, cls_token2), dim=1)
        self.last_layer.append(tr.detach().cpu().numpy())

        #NLP Concept: Probabilistic Model
        logits1 = self.classifier(tr.to(device))
        return logits1

def train_and_evaluate_ml_model(model, train, test):
    '''
    Train and evaluate a machine learning model.

    Args:
    - model (ML model): Machine learning model to be trained.
    - train (dict): Training data dictionary with 'embeds' and 'labels' keys.
    - test (dict): Test data dictionary with 'embeds' and 'labels' keys.

    Returns:
    - acc (float): Accuracy of the trained model.
    - f1 (float): Macro F1 score of the trained model.
    '''
    model.fit(train['embeds'], train['labels'])
    preds = model.predict(test['embeds'])
    acc = accuracy_score(test['labels'], preds)
    f1 = f1_score(test['labels'], preds, average='macro')
    return acc, f1

def train_and_test_ml_models(train, test, fused_feats=False, fused_feats_train=None, fused_feats_test=None):
    '''
    Train and test traditional machine learning models.

    Args:
    - train (dict): Training data dictionary with 'embeds' and 'labels' keys.
    - test (dict): Test data dictionary with 'embeds' and 'labels' keys.
    - fused_feats (bool): Flag indicating whether to use fused features.
    - fused_feats_train (numpy array): Fused features for training.
    - fused_feats_test (numpy array): Fused features for testing.
    '''
    models = [SVC(), RandomForestClassifier(), HistGradientBoostingClassifier(), AdaBoostClassifier()]

    for model in models:
        if fused_feats:
            acc, f1, = train_and_evaluate_ml_model(model, {'embeds': fused_feats_train, 'labels': train['labels']}, {'embeds': fused_feats_test, 'labels': test['labels']})
            print(f"ML Model Performance ({model.__class__.__name__} with Fused Feats) - Accuracy {acc}, F1: {f1}")
        else:
            acc, f1, = train_and_evaluate_ml_model(model, train, test)
            print(f"ML Model Performance ({model.__class__.__name__}) - Accuracy {acc}, F1: {f1}")

# Parameters
batch_size = 16
epochs = 5
learning_rate = 2e-5
weight_decay = 0.01

#NLP Concept: Transformers
MODEL_NAME1 = "distilbert/distilroberta-base"
MODEL_NAME2 = 'distilbert/distilbert-base-uncased'

# Tokenizers
tokenizer1 = AutoTokenizer.from_pretrained(MODEL_NAME1)
tokenizer2 = AutoTokenizer.from_pretrained(MODEL_NAME2)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(dataset['train']['text'], dataset['train']['label'], test_size=0.2, random_state=42, stratify=dataset['train']['label'])
train_data = {"text": X_train, "label": y_train}
test_data = {"text": X_test, "label": y_test}

# Convert into Dataset objects
dataset_train = Dataset.from_dict(train_data)
dataset_test = Dataset.from_dict(test_data)

# Tokenize and set format
tokenized_train1, tokenized_test1 = get_tokenized_train_test_set(dataset_train, dataset_test, tokenizer1)
tokenized_train2, tokenized_test2 = get_tokenized_train_test_set(dataset_train, dataset_test, tokenizer2)
tokenized_train1.set_format('torch')
tokenized_test1.set_format('torch')
tokenized_train2.set_format('torch')
tokenized_test2.set_format('torch')

# Create DataLoader
train_dataloader_model1 = DataLoader(tokenized_train1, batch_size=batch_size)
train_dataloader_model2 = DataLoader(tokenized_train2, batch_size=batch_size)
test_dataloader_model1 = DataLoader(tokenized_test1, batch_size=batch_size)
test_dataloader_model2 = DataLoader(tokenized_test2, batch_size=batch_size)

# Train and evaluate ensemble model
ensemble = EmotionEnsembleClassifier('distilroberta-base', 'distilbert-base-uncased', num_labels=6).to(device)
acc, f1, train_embeds, test_embeds = train_and_evaluate_model(ensemble, train_dataloader_model1, test_dataloader_model1, train_dataloader_model2, test_dataloader_model2, learning_rate, epochs, weight_decay)
print(f"Ensemble Model Performance: Accuracy:{acc['accuracy']}, Macro F1:{f1['f1']}")

# Extract embeddings for ML models
train_embeds = np.concatenate(ensemble.last_layer[-(2*(math.ceil(dataset_train.shape[0]/batch_size))):-(math.ceil(dataset_train.shape[0]/batch_size))])
eval_embeds = np.concatenate(ensemble.last_layer[-(math.ceil(dataset_train.shape[0]/batch_size)):])

# Prepare data for ML models
ml_train = {'embeds':train_embeds,'labels':dataset_train['label']}
ml_test = {'embeds':eval_embeds,'labels':dataset_test['label']}

# Train and test ML models
train_and_test_ml_models(ml_train, ml_test)
