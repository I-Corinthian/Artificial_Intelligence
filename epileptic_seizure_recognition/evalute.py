import torch
import pandas as pd
from transformers import BertForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.preprocessing import LabelEncoder
from datasets import Dataset

# Load the dataset
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

# Load the model and tokenizer
model = BertForSequenceClassification.from_pretrained('epileptic_seizure_recognition/model/eeg-bert')
tokenizer = AutoTokenizer.from_pretrained('epileptic_seizure_recognition/model/eeg-bert')

# Tokenize the data
encoded_data = tokenizer(data['text'].tolist(), padding=True, truncation=True, max_length=512, return_tensors='pt')
eval_dataset = Dataset.from_dict({
    'input_ids': encoded_data['input_ids'].numpy(), 
    'attention_mask': encoded_data['attention_mask'].numpy(),
    'labels': data['label'].values
})

# Define the compute_metrics function
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = torch.argmax(torch.tensor(logits), dim=-1).numpy()
    return {
        'accuracy': accuracy_score(labels, predictions),
        'precision': precision_score(labels, predictions, average='weighted'),
        'recall': recall_score(labels, predictions, average='weighted'),
        'f1': f1_score(labels, predictions, average='weighted')
    }

# Define the training arguments for evaluation
training_args = TrainingArguments(
    output_dir='epileptic_seizure_recognition/log',
    per_device_eval_batch_size=16,
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics
)

# Evaluate the model
metrics = trainer.evaluate()

# Print the results
print(f"Accuracy: {metrics.get('eval_accuracy', 'Not computed'):.4f}")
print(f"Precision: {metrics.get('eval_precision', 'Not computed'):.4f}")
print(f"Recall: {metrics.get('eval_recall', 'Not computed'):.4f}")
print(f"F1-Score: {metrics.get('eval_f1', 'Not computed'):.4f}")
print(f"Loss: {metrics.get('eval_loss', 'Not computed'):.4f}")

# Additional classification report
eval_predictions = trainer.predict(eval_dataset)
predictions = torch.argmax(torch.tensor(eval_predictions.predictions), dim=-1).numpy()
true_labels = eval_predictions.label_ids
report = classification_report(true_labels, predictions, target_names=label_encoder.classes_)
print("Classification Report:")
print(report)
