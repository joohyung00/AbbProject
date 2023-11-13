import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer 
import time
import datetime
import json

from tqdm import tqdm

from datasets import load_dataset 
from datasets import Dataset
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments
from transformers import TrainingArguments, Trainer
import numpy as np
import pyarrow as pa
import evaluate
import torch.nn as nn


    # dataset = load_dataset("yelp_review_full")
    # print(type(dataset))
    # print(type(dataset["train"]))
    # print(type(dataset["train"][0]))
    # print((dataset["train"][0]))
    # print()
print("[Loading tokenizer...]")
dataset = load_dataset("json", data_files = "kick_train.json")
print("[Loading tokenizer complete]")


    # tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
print("[Loading tokenizer...]")
tokenizer = AutoTokenizer.from_pretrained(
        "KT-AI/midm-bitext-S-7B-inst-v1",
        trust_remote_code = True
    )
print("[Loading tokenizer complete]")


# dummy_data = "### User; I see a korean report of a chemical industry, and some text field says '개스킷 교체 필요함' in a container of a chemical container. What does this mean, and what should I do? \n ### Midm;"
    
# start = time.time()
# data = tokenizer(dummy_data, return_tensors = "pt")
# end = time.time()
# sec = (end - start)
# result = datetime.timedelta(seconds = sec)
# print("Tokenizing Time: ", result)

# print(data)

def tokenize_function(examples):
    # return tokenizer(examples["text"], padding = "max_length", truncation = True)
    return tokenizer(examples["input"], return_tensors = "pt", padding = "longest")

print("[Tokenizing dataset...]")
tokenized_dataset = dataset.map(tokenize_function, batched = True)
print("[Tokenizing dataset complete]")


print("[Sampling dataset...]")
# train_dataset = tokenized_dataset["train"].shuffle(seed = 42).select( range( int(len(tokenized_dataset["train"]) * 0.9) ) )
# eval_range = range( int(len(tokenized_dataset["train"]) * 0.1) - 1 )
# eval_range = [ i + int(len(tokenized_dataset["train"]) * 0.9) for i in eval_range ]
# eval_dataset = tokenized_dataset["train"].shuffle(seed = 42).select( eval_range )
train_dataset = tokenized_dataset["train"].shuffle(seed = 42).select( range( 10000 ) )
eval_range = [ i + 10000 for i in range(1000) ]
eval_dataset = tokenized_dataset["train"].shuffle(seed = 42).select( eval_range )
print("[Sampling dataset complete]")

print("\t", type(train_dataset))
print("\t", train_dataset.shape)
print("\t", type(eval_dataset))
print("\t", eval_dataset.shape)

# small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
# small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(1000))

# print(type(small_train_dataset))
# print(small_train_dataset.shape)
# print(small_train_dataset[0])


    # model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=5)
print("[Loading model...]")
model = AutoModelForCausalLM.from_pretrained(
        "KT-AI/midm-bitext-S-7B-inst-v1", 
        trust_remote_code = True
    )
print("[Loading model complete]")

print("[Model cuda...]")
model.cuda()
print("[Model cuda complete]")

print("[Model train...]")
model.train()
print("[Model train complete]")


    # training_args = TrainingArguments(output_dir = "/root/fine_tuned_midm_checkpoints")
training_args = TrainingArguments(output_dir = "/root/fine_tuned_midm_checkpoints", evaluation_strategy = "epoch")

# Evaluate
    # metric = evaluate.load("accuracy")
    # def compute_metrics(eval_pred):
    #     logits, labels = eval_pred
    #     predictions = np.argmax(logits, axis = -1)
        
    #     return metric.compute(predictions = predictions, references=labels)
loss_fn = nn.CrossEntropyLoss(reduction = "sum")
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    loss = loss_fn(logits, labels)
    
    return loss




    # trainer = Trainer(
    #     model = model,
    #     args = training_args,
        
    #     train_dataset = small_train_dataset,
    #     eval_dataset = small_eval_dataset,
    #     compute_metrics = compute_metrics
    # )

print("[Trainer definition...]")
trainer = Trainer(
    model = model,
    args = training_args,
    
    train_dataset = train_dataset,
    eval_dataset = eval_dataset,
    compute_metrics = compute_metrics
)
print("[Trainer definition complete]")

print("[Training...]")
trainer.train()
print("[Training complete]")

exit()