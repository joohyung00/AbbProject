import pandas as pd
import json
import random



def translateToString(df):

    instruction = "임의의 구인 광고가 '사기'인지 '진실'인지 판단하고 싶다. 다음은 구인 광고에 대한 정보를 통해 해당 구인 광고가 거짓인지 사실인지 답하여라 ('사기' 또는 '진실'로만 답하여라). \n\n"

    positives = []
    negatives = []

    # count = 5
    for i in range(df.shape[0]):
        row = df.iloc[[i]]

        translated_string = ""
        
        # Text
        # translated_string += "본 광고는 " + str(row.iloc[0]["title"]) + "란 직업의 광고이며, 구체적인 설명은"
        # if row.iloc[0]["description"] != None: 
        #     translated_string += " 다음과 같다: " + str(row.iloc[0]["description"]) + " "
        # else:
        #     translated_string += " 적혀 있지 않다. "
        # if row.iloc[0]["required_experience"] != None:
        #     translated_string += "해당 직업을 위해 필요한 경험은 " + str(row.iloc[0]["required_experience"]) + " 이며, "
        # else:
        #     translated_string += "해당 직업을 위해 필요한 경험은 적혀 있지 않으며, "
        # if row.iloc[0]["required_education"] != None:
        #     translated_string += "필요 교육 이력은 " + str(row.iloc[0]["required_education"]) + " 이다. "
        # else:
        #     translated_string += "필요 교육 이력은 적혀 있지 않다. "
        
        # Numerical
        if row.iloc[0]["salary_range"] != None:
            translated_string += "본 광고의 직업의 연봉의 범위는 " + str(row.iloc[0]["salary_range"]) + " 달러이다."
        else:
            translated_string += "본 광고의 직업의 연봉은 명시되어 있지 않다."
        
        input = translated_string
        
        label = row.iloc[0]["fraudulent"]
        if label == 1:
            output = "사기"
        else:
            output = "진실"

        # print(instruction)
        # print(input)
        # print(output)
        # print()

        # if label not in label_count:
        #     label_count[label] = 0
        # label_count[label] += 1

        data = {
            "instruction": instruction,
            "input": input,
            "output": output
        }
        
        if output == "사기":
            positives.append(data)
        else:
            negatives.append(data)

    # Sample 1/3 of successes
    # sampled_successes = random.sample(successes, int(len(successes)/3))

    train_dataset = []
    
    # positive_rate = len(positives) / (len(positives) + len(negatives))
    # negative_rate = len(negatives) / (len(positives) + len(negatives))
    
    # Train dataset
    train_dataset = positives + negatives

    print()
    print(len(train_dataset))
    train_count = {}
    for data in train_dataset:
        if data["output"] not in train_count:
            train_count[data["output"]] = 0
        train_count[data["output"]] += 1
    print(train_count)

    # with open("fake_numerical_train.json", "w") as file:
    #     json.dump(train_dataset, file)
    with open("fake_numerical_test.json", "w") as file:
        json.dump(train_dataset, file)

    return

# df = pd.read_csv('fake_train.csv')
df = pd.read_csv('fake_test.csv')

translateToString(df)