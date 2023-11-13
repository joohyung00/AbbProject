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

with open("kick_train.json", "r") as file:
    dataset_json = json.load(file)

train_length = int(len(dataset_json) * 0.9) - 1
eval_length = int(len(dataset_json) * 0.1) - 1

# train_dataset = dataset_json[:train_length]
# eval_dataset = dataset_json[train_length:train_length + eval_length]

train_dataset = dataset_json[:18000]
eval_dataset = dataset_json[18000:20000]



print(len(train_dataset))
print(len(eval_dataset))

    # tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
print("[Loading tokenizer...]")
tokenizer = AutoTokenizer.from_pretrained(
        "KT-AI/midm-bitext-S-7B-inst-v1",
        trust_remote_code = True
    )
print("[Loading tokenizer complete]")

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
model.eval()
print("[Model train complete]")

def rfind(list_, target):
    index = None
    for i, element in enumerate(list_):
        if element == target:
            index = i
    return index

def find(list_, target):
    for i, element in enumerate(list_):
        if element == target:
            return i
    return None
    

def extractSubstring(input_list):
    start_index = rfind(input_list, ";") + 1
    end_index = rfind(input_list, "</s>")
    return input_list[start_index : end_index]


def predictWithPrompt(prompt, data, label):

    anrg = prompt + data
    question = f"###User;{anrg}\n###Midm;"
    data = tokenizer(question, return_tensors = "pt")
    
    # streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    pred = model.generate(
        input_ids = data.input_ids[..., : -1].cuda(),
        # streamer = streamer,
        use_cache = True,
        max_new_tokens = float('inf')
    )
    decoded_text = tokenizer.batch_decode(pred[0], skip_special_tokens = True)

    # between ; and </s>
    # print(decoded_text)
    extract_substring_list = extractSubstring(decoded_text)
    # print(extract_substring_list)
    prediction = " ".join(extract_substring_list)

    if label == "성공" and "실패" not in prediction:
        return True
    if label == "실패" and "실패" in prediction:
        return True
    return False

# prompt = "다음은 임의의 프로젝트에 대한 정보이다. 프로젝트의 정보를 바탕으로 해당 프로젝트가 성공하였는지, \
# 또는 실패하였는지 예상하라('성공' 또는 '실패'로만 대답하라). \n"
# anrg = "본 프로젝트의 이름은 LOS ANGELES KOREA TOWN MURAL PROJECT이다. \
# 프로젝트의 키워드는 los-angeles-korea-town-mural-project이며, 구체적인 설명은 다음과 같다 - The LA Korea Town Mural Project \
# would create a mural of traditional Korean culture and history in the Los Angeles Korea Town area. 프로젝트 책임자의 국적은 \
# US이며, 목표 금액은 30000.0 USD이다. 본 프로젝트의 런칭 시각은 1356654814이며, due date는 1359246814이다. 본 프로젝트는 지지자들의 \
# 수는 5 명이며, 지지자들과의 온라인 커뮤니케이션을 허용하지 않았다."

# predictNoShot(prompt, anrg, "실패")



# NO SHOT

true_positive = 0
true_negative = 0
false_positive = 0
false_negative = 0

for data in tqdm(eval_dataset):
    prompt = data["instruction"]
    info = data["input"]
    label = data["output"]
    result = predictWithPrompt(prompt, info, label)

    if label == "성공":
        if result: true_positive += 1
        else:      false_negative += 1
    else:
        if result: true_negative += 1
        else:      false_positive += 1

tp = true_positive
tn = true_negative
fp = false_positive
fn = false_negative

accuracy = (tp + tn) / (tp + tn + fp + fn)
recall = tp / (tp + fn)
precision = tp / (tp + fp)

with open("prompting_result.txt", "w") as file:
    print("Accuracy: ", accuracy)
    print("Recall: ", recall)
    print("Precision: ", precision)
    result = {"Attempt": "Zero Shot", "Accuracy": accuracy, "Recall": recall, "Precision": precision}
    json.dump(result, file)

# FULL SHOT
true_positive = 0
true_negative = 0
false_positive = 0
false_negative = 0


i = -1
for data in tqdm(eval_dataset):
    prompt = "어떠한 펀딩 프로젝트가 펀딩에 성공하였는지, 또는 실패하였는지에 대하여 예측하고자 한다. 이에 대한 우리가 알고 있는 예시 3개는 다음과 같다. "
    for _ in range(3):
        i += 1
        prompt += "\n - '내용': "
        prompt += train_dataset[i]["input"]
        prompt += ", '결과': "
        prompt += train_dataset[i]["output"]
    
    prompt = "\n\n 위의 결과들을 이용하여, 다음으로 주어지는 정보의 프로젝트의 프로젝트가 성공하였는지, 또는 실패하였는지 예상하라('성공' 또는 '실패'로만 대답하라)."
    info = "\n\n" + data["input"]
    label = data["output"]
    
    result = predictWithPrompt(prompt, info, label)

    if label == "성공":
        if result: true_positive += 1
        else:      false_negative += 1
    else:
        if result: true_negative += 1
        else:      false_positive += 1

tp = true_positive
tn = true_negative
fp = false_positive
fn = false_negative

accuracy = (tp + tn) / (tp + tn + fp + fn)
recall = tp / (tp + fn)
precision = tp / (tp + fp)

with open("prompting_result.txt", "a") as file:
    print("Accuracy: ", accuracy)
    print("Recall: ", recall)
    print("Precision: ", precision)
    result = {"Attempt": "Full Shot", "Accuracy": accuracy, "Recall": recall, "Precision": precision}
    file.write(json.dumps(result))
    