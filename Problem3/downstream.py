import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import T5EncoderModel, AutoTokenizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import pandas as pd
import numpy as np

# Load the dataset from the Excel file
file_path = '/content/P3 - sample dataset.xlsx'  # Replace with the correct path
df = pd.read_excel(file_path)

# Extract the input texts and labels
texts = df['source code'].tolist()
labels = df[['Reentrancy Attack', 'Unsafe Delegate calls', 'Phishing Through tx.origin', 
             'Block Timestamp Manipulation', 'Ownership Exploit', 'Private Data Exploit']].astype(int).values

# Define a dataset class
class MultiLabelDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.float)
        }

# Define the multi-label classifier
class MultiLabelClassifier(nn.Module):
    def __init__(self, encoder, num_labels, hidden_size=1024):
        super().__init__()
        self.encoder = encoder
        self.classifier = nn.Linear(hidden_size, num_labels)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, input_ids, attention_mask):
        encoder_outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = encoder_outputs.last_hidden_state
        pooled_output = last_hidden_state[:, 0, :]  # Use the first token's representation
        logits = self.classifier(pooled_output)
        output = self.sigmoid(logits)
        return output

# Load the T5 encoder model and tokenizer
model_name = "hululuzhu/solidity-t5"
encoder = T5EncoderModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Split the dataset into training and testing sets
train_texts, test_texts, train_labels, test_labels = train_test_split(texts, labels, test_size=0.2, random_state=42)

# Create datasets and dataloaders
train_dataset = MultiLabelDataset(train_texts, train_labels, tokenizer)
test_dataset = MultiLabelDataset(test_texts, test_labels, tokenizer)

train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=2)

# Initialize the model, loss function, and optimizer
num_labels = train_labels.shape[1]  # Number of labels in your dataset
model = MultiLabelClassifier(encoder, num_labels)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)

# Training loop
epochs = 3  # Adjust the number of epochs as needed

for epoch in range(epochs):
    model.train()
    total_loss = 0
    
    for batch in train_dataloader:
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
        
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = criterion(outputs, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    avg_train_loss = total_loss / len(train_dataloader)
    print(f"Epoch {epoch+1}/{epochs} - Training Loss: {avg_train_loss}")

# Evaluation
model.eval()
all_preds = []
all_labels = []

with torch.no_grad():
    for batch in test_dataloader:
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
        
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        preds = (outputs > 0.5).float()  # Threshold predictions at 0.5
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

all_preds = np.array(all_preds)
all_labels = np.array(all_labels)

# Calculate metrics
accuracy = accuracy_score(all_labels, all_preds)
f1 = f1_score(all_labels, all_preds, average='micro')

print(f"Accuracy: {accuracy}")
print(f"F1 Score: {f1}")
