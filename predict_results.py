import json
from sklearn.metrics import roc_auc_score, recall_score, precision_score, accuracy_score


count = {}

preds = []
labels = []
with open("predict_results.jsonl", "r") as file:
    for line in file:
        result = json.loads(line)
        
        prediction = result["predict"]
        if "사기" in prediction: prediction = 1
        else: prediction = 0
        
        label = result["label"]
        if "사기" in label: label = 1
        else: label = 0
        
        preds.append(prediction)
        labels.append(label)
        
        if result["label"] not in count:
            count[result["label"]] = 0
        count[result["label"]] += 1
        
print("AUC: ", roc_auc_score(labels, preds))
print("Accuracy: ", accuracy_score(labels, preds))
print("Recall: ", recall_score(labels, preds))
print("Precision: ", precision_score(labels, preds))
print(count)