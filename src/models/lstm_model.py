"""LSTM model for text classification."""
import torch
import torch.nn as nn
from src.config import DEVICE, LSTM_MAX_LEN

class CyberbullyingLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes,
                 num_layers=2, bidirectional=True, dropout=0.5,
                 pretrained_embeddings=None, freeze_embeddings=False):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        
        if pretrained_embeddings is not None:
            self.embedding.weight.data.copy_(pretrained_embeddings)
            if freeze_embeddings:
                self.embedding.weight.requires_grad = False
        
        self.lstm = nn.LSTM(
            embed_dim, hidden_dim, num_layers=num_layers,
            bidirectional=bidirectional, batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(lstm_output_dim, num_classes)
    
    def forward(self, x):
        # x: (batch_size, seq_len)
        embedded = self.embedding(x)
        # embedded: (batch_size, seq_len, embed_dim)
        lstm_out, (hidden, cell) = self.lstm(embedded)
        # lstm_out: (batch_size, seq_len, hidden_dim * num_directions)
        # hidden: (num_layers * num_directions, batch_size, hidden_dim)
        
        # Take final hidden state
        if self.lstm.bidirectional:
            hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
        else:
            hidden = hidden[-1]
        
        hidden = self.dropout(hidden)
        logits = self.fc(hidden)
        return logits


class LSTMWrapper:
    """Wrapper for inference with CyberbullyingLSTM."""
    def __init__(self, model, vocab, labels, batch_size=256):
        self.model = model
        self.vocab = vocab
        self.label2idx = {lbl: i for i, lbl in enumerate(labels)}
        self.inv_label_map = {i: lbl for i, lbl in enumerate(labels)}
        self.batch_size = batch_size
        self.model.eval()
    
    def predict(self, texts):
        all_preds = []
        with torch.no_grad():
            for i in range(0, len(texts), self.batch_size):
                batch = texts[i:i+self.batch_size]
                indices = []
                for text in batch:
                    tokens = text.split()
                    idxs = [self.vocab.get(t, self.vocab["<UNK>"]) for t in tokens]
                    if len(idxs) < LSTM_MAX_LEN:
                        idxs += [self.vocab["<PAD>"]] * (LSTM_MAX_LEN - len(idxs))
                    else:
                        idxs = idxs[:LSTM_MAX_LEN]
                    indices.append(idxs)
                
                x = torch.tensor(indices, dtype=torch.long).to(DEVICE)
                outputs = self.model(x)
                all_preds.extend(outputs.argmax(dim=1).cpu().tolist())
        return all_preds
