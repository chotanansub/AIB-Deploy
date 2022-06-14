from transformers import AutoTokenizer, AutoModelForSequenceClassification

def load_wangchan():
    global model 
    global tokenizer
    tokenizer = AutoTokenizer.from_pretrained("CH0KUN/autotrain-TNC_Domain_WangchanBERTa-921730254")
    model = AutoModelForSequenceClassification.from_pretrained("CH0KUN/autotrain-TNC_Domain_WangchanBERTa-921730254")

def all_preprocessing(text):
    input = tokenizer(text,return_tensors="pt")
    outputs =  model(**input)
    output_LST = outputs.logits.softmax(dim=-1).tolist()[0]
    domainProb = max(output_LST) 
    domainIndex = output_LST.index(domainProb)
    return domainIndex, domainProb

def is_model_ready():
    try:
        all_preprocessing("สวัสดีครับ")
        return True
    except:
        return False
        


