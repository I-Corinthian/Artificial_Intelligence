import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from datasets import Dataset


df = pd.read_csv('epileptic_seizure_recognition\dataset\Epileptic_Seizure_Recognition.csv', header=None, dtype=str)

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

train_df, test_df = train_test_split(data[['text', 'label']], test_size=0.2, random_state=42)

train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
model = AutoModelForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

def tokenize_function(example):
    return tokenizer(example['text'], padding="max_length", truncation=True)

tokenized_train = train_dataset.map(tokenize_function, batched=True)
tokenized_test = test_dataset.map(tokenize_function, batched=True)

training_args = TrainingArguments(
    output_dir='epileptic_seizure_recognition\log',
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='epileptic_seizure_recognition\log\logs',
    logging_steps=10,
    eval_strategy="epoch",
    save_total_limit=1,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test,
)

trainer.train()

model.save_pretrained('epileptic_seizure_recognition/model/eeg-bert')
tokenizer.save_pretrained('epileptic_seizure_recognition/model/eeg-bert')


