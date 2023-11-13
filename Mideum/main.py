import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer 
import time
import datetime


def main():

    start = time.time()
    tokenizer = AutoTokenizer.from_pretrained(
        "KT-AI/midm-bitext-S-7B-inst-v1",
        trust_remote_code = True
    )
    end = time.time()
    sec = (end - start)
    result = datetime.timedelta(seconds=sec)
    print("Load Tokenizer: ", result)


    dummy_data = "### User; I see a korean report of a chemical industry, and some text field says '개스킷 교체 필요함' in a container of a chemical container. What does this mean, and what should I do? 엄준식 엄준식 엄준식 엄준식 엄준식 엄준식 엄준식 엄준식 엄준식 엄준식 엄준식 엄준식 엄준식 엄준식 엄준식 엄준식 엄준식 엄준식 엄준식 엄준식 엄준식 엄준식 엄준식 엄준식 엄준식 엄준식 엄준식 엄준식 엄준식 엄준식 엄준식   \n ### Midm;"
    

    start = time.time()
    data = tokenizer(dummy_data, return_tensors = "pt")
    end = time.time()
    sec = (end - start)
    result = datetime.timedelta(seconds = sec)
    print("Tokenizing Time: ", result)
    
    print(data)

    start = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        "KT-AI/midm-bitext-S-7B-inst-v1", 
        trust_remote_code=True
    )
    end = time.time()
    sec = (end - start)
    result = datetime.timedelta(seconds=sec)
    print("Load Model: ", result)


    start = time.time()
    model.cuda()
    end = time.time()
    sec = (end - start)
    result = datetime.timedelta(seconds = sec)
    print("Cuda Loading: ", result)


    start = time.time()
    model.eval()
    end = time.time()
    sec = (end - start)
    result = datetime.timedelta(seconds = sec)
    print("Model Eval Mode: ", result)


    


    start = time.time()
    streamer = TextStreamer(tokenizer, skip_prompt = True, skip_special_tokens = True)
    end = time.time()
    sec = (end - start)
    result = datetime.timedelta(seconds = sec)
    print("Streamer Definition: ", result)


    start = time.time()
    pred = model.generate(
        input_ids = data.input_ids[..., :-1].cuda(),
        streamer = streamer,
        use_cache = True,
        max_new_tokens = float('inf')
    )
    end = time.time()
    sec = (end - start)
    result = datetime.timedelta(seconds = sec)
    print("Prediction: ", result)

    decoded_text = tokenizer.batch_decode(pred[0], skip_special_tokens = True)

    return

if __name__ == "__main__":

    main()