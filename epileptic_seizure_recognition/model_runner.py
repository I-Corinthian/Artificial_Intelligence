from transformers import AutoTokenizer, BertForSequenceClassification
import torch
from sklearn.calibration import LabelEncoder
import torch
import pandas as pd

model = BertForSequenceClassification.from_pretrained('epileptic_seizure_recognition\model\eeg-bert')
tokenizer = AutoTokenizer.from_pretrained('epileptic_seizure_recognition\model\eeg-bert')

def prd(data):
    data = pd.DataFrame(data)
    data = data.astype(str)
    data['text'] = data.apply(lambda row: ' '.join(row))
    text = data['text'][0]
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=1).item()
    return predicted_class