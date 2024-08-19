import os
import torch
import pandas as pd
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertModel, AdamW, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# This is so I don't have to keep uploading on Colab.
import requests
from requests.auth import HTTPBasicAuth

class BERTClassifier(nn.Module):
  def __init__(self, bert_model_name, num_classes):
      super(BERTClassifier, self).__init__()
      self.bert = BertModel.from_pretrained(bert_model_name)
      self.dropout = nn.Dropout(0.1)
      self.fc = nn.Linear(self.bert.config.hidden_size, num_classes)

  def forward(self, input_ids, attention_mask):
      outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
      pooled_output = outputs.pooler_output
      x = self.dropout(pooled_output)
      logits = self.fc(x)
      return logits
  
class TextClassificationDataset(Dataset):
  def __init__(self, texts, labels, tokenizer, max_length):
          self.texts = texts
          self.labels = labels
          self.tokenizer = tokenizer
          self.max_length = max_length
  def __len__(self):
      return len(self.texts)
  def __getitem__(self, idx):
      text = str(self.texts[idx])
      label = self.labels[idx]
      encoding = self.tokenizer(text, return_tensors='pt', max_length=self.max_length, padding='max_length', truncation=True)
      return {'input_ids': encoding['input_ids'].flatten(), 'attention_mask': encoding['attention_mask'].flatten(), 'label': torch.tensor(label-1)}
  
def train(model, data_loader, optimizer, scheduler, device):
  model.train()
  for i,batch in enumerate(data_loader):
      optimizer.zero_grad()
      input_ids = batch['input_ids'].to(device)
      attention_mask = batch['attention_mask'].to(device)
      labels = batch['label'].to(device)
      outputs = model(input_ids, attention_mask)
      loss = nn.CrossEntropyLoss()(outputs, labels)
      if i % 100 == 0:
        print(f"Batch: {i}")
      loss.backward()
      optimizer.step()
      scheduler.step()

def evaluate(model, data_loader, device):
    model.eval()
    predictions = []
    actual_labels = []
    with torch.no_grad():
        for batch in data_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            _, preds = torch.max(outputs, dim=1)
            predictions.extend(preds.cpu().tolist())
            actual_labels.extend(labels.cpu().tolist())
    return accuracy_score(actual_labels, predictions), classification_report(actual_labels, predictions)

def predict_sentiment(text, model, tokenizer, device, max_length=128):
    model.eval()
    encoding = tokenizer(text, return_tensors='pt', max_length=max_length, padding='max_length', truncation=True)
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)

    with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            _, preds = torch.max(outputs, dim=1)
    return preds.item()

df = pd.read_csv('./data/train.csv')
df_test = pd.read_csv('./data/test.csv')
df_test['Review'] = df_test['Review'].astype(str)
texts = df['Review'].tolist()
labels = df['overall'].tolist()

bert_model_name = 'bert-large-uncased'
num_classes = 5
max_length = 128
batch_size = 128
num_epochs = 10
learning_rate = 2e-5

train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, test_size=0.2, random_state=42)
#Training on the entire dataset while keeping the validation same.
train_texts = texts
train_labels = labels

tokenizer = BertTokenizer.from_pretrained(bert_model_name)
train_dataset = TextClassificationDataset(train_texts, train_labels, tokenizer, max_length)
val_dataset = TextClassificationDataset(val_texts, val_labels, tokenizer, max_length)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BERTClassifier(bert_model_name, num_classes).to(device)
# If you are doing the load and train thing, use this to load:
# model.load_state_dict(torch.load(f"/bert{epoch}e.pt"))
optimizer = AdamW(model.parameters(), lr=learning_rate)
total_steps = len(train_dataloader) * num_epochs
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}/{num_epochs}")
    train(model, train_dataloader, optimizer, scheduler, device)
    accuracy, report = evaluate(model, val_dataloader, device)
    print(f"Validation Accuracy: {accuracy:.4f}")
    print(report)
    torch.save(model.state_dict(), f"./bertne{epoch}e.pt")
    df_submission = pd.DataFrame()
    df_submission['id'] = df_test['id']
    for index, row in df_test.iterrows():
        value = predict_sentiment(row['Review'], model, tokenizer, device)
        df_submission.at[index, 'overall'] = value
    df_submission.to_csv(f"./submissionBertNE{epoch + 1}Epochs.csv", index = False)

