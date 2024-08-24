from transformers import AutoTokenizer, BertForSequenceClassification
import torch
from sklearn.calibration import LabelEncoder
import torch
import pandas as pd


df = pd.read_csv('epileptic_seizure_recognition/dataset/Epileptic_Seizure_Recognition.csv', header=None, dtype=str)

df[179] = df[179].apply(lambda x: 1 if x == "1" else 0)


df = df.drop(df.index[0])
df = df.drop(columns=df.columns[0])
df = df.astype(str)

X = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

data = pd.DataFrame(X)
data['text'] = data.apply(lambda row: ' '.join(row), axis=1)
label_encoder = LabelEncoder()
y_int = label_encoder.fit_transform(y)
data['label'] = y_int

model = BertForSequenceClassification.from_pretrained('epileptic_seizure_recognition\model\eeg-bert')
tokenizer = AutoTokenizer.from_pretrained('epileptic_seizure_recognition\model\eeg-bert')

# data for prediction
text = data['text'][33] #INPUT
inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)


with torch.no_grad():
    outputs = model(**inputs)

logits = outputs.logits
predicted_class = torch.argmax(logits, dim=1).item()

print(f"Predicted class: {predicted_class}")
