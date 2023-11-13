import pandas as pd
import json
import random





def translateToString(df):

    instruction = "다음은 임의의 프로젝트에 대한 정보이다. 프로젝트의 정보를 바탕으로 해당 프로젝트가 '성공'하였는지, 또는 '실패'하였는지 예상하라. '성공' 또는 '실패'로만 대답하라."

    successes = []
    fails = []


    # count = 5
    for i in range(df.shape[0]):
        row = df.iloc[[i]]

        # if count < 0:
        #     break
        # count -= 1

        translated_string = ""
        
        # Text
        translated_string += "본 프로젝트의 이름은 " + str(row.iloc[0]["name"]) + "이다. "
        translated_string += "프로젝트의 키워드는 " + str(row.iloc[0]["keywords"]) + "이며, "
        translated_string += "구체적인 설명은 다음과 같다 - " + str(row.iloc[0]["desc"]) + " "
        
        # Numerical
        translated_string += "프로젝트 책임자의 국적은 " + str(row.iloc[0]["country"]) + "이며, "
        translated_string += "목표 금액은 " + str(row.iloc[0]["goal"]) + " " + str(row.iloc[0]["currency"]) + "이다. "
        translated_string += "본 프로젝트의 런칭 시각은 " + str(row.iloc[0]["launched_at"]) + "이며, due date는 " + str(row.iloc[0]["deadline"]) + "이다. "
        translated_string += "본 프로젝트는 지지자들의 수는 " + str(row.iloc[0]["backers_count"]) + " 명이며, 지지자들과의 온라인 커뮤니케이션을 "
        if row.iloc[0]["disable_communication"] == True:
            translated_string += "허용하였다."
        else:
            translated_string += "허용하지 않았다."


        input = translated_string
        
        label = row.iloc[0]["final_status"]
        
        if label == 1:
            output = "성공"
        else:
            output = "실패"

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
        
        if output == "성공":
            successes.append(data)
        else:
            fails.append(data)

    # Sample 1/3 of successes
    # sampled_successes = random.sample(successes, int(len(successes)/3))

    train_dataset = []
    test_dataset = []
    
    fail_rate = len(fails) / (len(successes) + len(fails))
    success_rate = len(successes) / (len(successes) + len(fails))
    
    # Train dataset
    TRAIN_NUM = 20000
    train_fails     = fails[:int(TRAIN_NUM * fail_rate)]
    train_successes = successes[:int(TRAIN_NUM * success_rate)]
    train_dataset = train_fails + train_successes
    random.shuffle(train_dataset)

    # Test dataset
    TEST_NUM = 2000
    test_fails     = fails[int(TRAIN_NUM * fail_rate) : int(TRAIN_NUM * fail_rate) + int(TEST_NUM * fail_rate)]
    test_successes = successes[int(TRAIN_NUM * success_rate) : int(TRAIN_NUM * success_rate) + int(TEST_NUM * success_rate)]
    test_dataset = test_successes + test_fails
    random.shuffle(test_dataset)
    
    print()
    print(len(train_dataset))
    train_count = {}
    for data in train_dataset:
        if data["output"] not in train_count:
            train_count[data["output"]] = 0
        train_count[data["output"]] += 1
    print(train_count)
    
    print()
    print(len(test_dataset))
    test_count = {}
    for data in test_dataset:
        if data["output"] not in test_count:
            test_count[data["output"]] = 0
        test_count[data["output"]] += 1
    print(test_count)
    print()


    with open("kick_original_naive_20000.json", "w") as file:
        json.dump(train_dataset, file)

    with open("kick_original_naive_2000.json", "w") as file:
        json.dump(test_dataset, file)

    return

df = pd.read_csv('kick_train.csv')

translateToString(df)