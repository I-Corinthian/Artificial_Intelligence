from sklearn.calibration import LabelEncoder
import torch
import pandas as pd
from transformers import BertForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from datasets import Dataset

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

model = BertForSequenceClassification.from_pretrained('epileptic_seizure_recognition/model/eeg-bert')
tokenizer = AutoTokenizer.from_pretrained('epileptic_seizure_recognition/model/eeg-bert', num_labels=2)


encoded_data = tokenizer(data['text'].tolist(), padding=True, truncation=True, max_length=512, return_tensors='pt')

eval_dataset = Dataset.from_dict({
    'input_ids': encoded_data['input_ids'].numpy(), 
    'attention_mask': encoded_data['attention_mask'].numpy(),
    'labels': data['label'].values
})


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = torch.argmax(torch.tensor(logits), dim=-1)
    return {
        'accuracy': (predictions.numpy() == labels).mean()
    }

training_args = TrainingArguments(
    output_dir='epileptic_seizure_recognition/log',
    per_device_eval_batch_size=16,
)

trainer = Trainer(
    model=model,
    args=training_args,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics
)

metrics = trainer.evaluate()

# Print the results
print(f"Accuracy: {metrics.get('eval_accuracy', 'Not computed')}")
print(f"Loss: {metrics.get('eval_loss', 'Not computed')}")
