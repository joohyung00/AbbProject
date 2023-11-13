from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import TrainingArguments
import numpy as np
import evaluate
from transformers import TrainingArguments, Trainer


# 1. Get data 
dataset = load_dataset("yelp_review_full")
dataset["train"][100]

print(type(dataset))
print(dataset.shape)

# 2. Get tokenizer
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

    # 2.1. Tokenize dataset
tokenized_datasets = dataset.map(tokenize_function, batched=True)

print(type(tokenized_datasets))
print(tokenized_datasets.shape)

small_train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(1000))
small_eval_dataset = tokenized_datasets["test"].shuffle(seed=42).select(range(1000))

print(type(small_train_dataset))
print(small_train_dataset.shape)

exit()

# 3. Load model
model = AutoModelForSequenceClassification.from_pretrained("bert-base-cased", num_labels=5)


# 4. Set training arguments
training_args = TrainingArguments(output_dir="test_trainer")
training_args = TrainingArguments(output_dir="test_trainer", evaluation_strategy="epoch")

# 5. Set training goal
metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


# 6. Train
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=small_train_dataset,
    eval_dataset=small_eval_dataset,
    compute_metrics=compute_metrics,
)
trainer.train()