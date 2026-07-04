# training/train.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import os
import json
from datetime import datetime
from pathlib import Path

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    # Model
    vocab_size = 32000
    max_seq_length = 4096
    hidden_size = 4096
    num_layers = 32
    num_heads = 32
    ffn_hidden_size = 11008
    dropout = 0.1
    
    # Training
    batch_size = 8
    gradient_accumulation_steps = 4
    learning_rate = 5e-4
    num_epochs = 3
    warmup_steps = 1000
    max_steps = None
    
    # Checkpointing
    checkpoint_dir = "checkpoints"
    save_every = 1000
    eval_every = 500
    
    # Data
    data_dir = "data/processed"
    train_file = "cybersecurity_train.jsonl"
    val_file = "cybersecurity_val.jsonl"
    
    # Hardware
    device = "cuda" if torch.cuda.is_available() else "cpu"
    num_gpus = torch.cuda.device_count()
    mixed_precision = True
    gradient_checkpointing = True
    
    # Logging
    log_dir = "logs"
    log_every = 10


# ============================================================================
# DATASET
# ============================================================================

class CybersecurityDataset(Dataset):
    """Load cybersecurity training data from JSONL files."""
    
    def __init__(self, file_path, tokenizer, max_length=4096):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []
        
        print(f"Loading dataset from {file_path}...")
        with open(file_path, 'r') as f:
            for line in f:
                if line.strip():
                    self.data.append(json.loads(line))
        
        print(f"Loaded {len(self.data)} examples")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Combine text fields (adjust based on your data structure)
        text = item.get('text', '')
        
        # Tokenize
        tokens = self.tokenizer.encode(text, add_special_tokens=True)
        
        # Truncate or pad
        if len(tokens) > self.max_length:
            tokens = tokens[:self.max_length]
        else:
            tokens = tokens + [self.tokenizer.pad_token_id] * (self.max_length - len(tokens))
        
        input_ids = torch.tensor(tokens[:-1], dtype=torch.long)
        labels = torch.tensor(tokens[1:], dtype=torch.long)
        
        return {
            'input_ids': input_ids,
            'labels': labels,
            'attention_mask': (input_ids != self.tokenizer.pad_token_id).long()
        }


# ============================================================================
# MODEL (Simplified Transformer)
# ============================================================================

class TransformerModel(nn.Module):
    """Simplified transformer-based language model for cybersecurity."""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.positional_encoding = nn.Embedding(config.max_seq_length, config.hidden_size)
        
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=config.hidden_size,
                nhead=config.num_heads,
                dim_feedforward=config.ffn_hidden_size,
                dropout=config.dropout,
                batch_first=True
            )
            for _ in range(config.num_layers)
        ])
        
        self.norm = nn.LayerNorm(config.hidden_size)
        self.head = nn.Linear(config.hidden_size, config.vocab_size)
    
    def forward(self, input_ids, attention_mask=None):
        seq_length = input_ids.shape[1]
        
        # Embeddings
        x = self.embedding(input_ids)
        pos_ids = torch.arange(seq_length, device=input_ids.device).unsqueeze(0)
        x = x + self.positional_encoding(pos_ids)
        
        # Create attention mask
        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attention_mask = (1.0 - attention_mask) * -10000.0
        
        # Transformer layers
        for layer in self.layers:
            x = layer(x, src_key_padding_mask=attention_mask)
        
        x = self.norm(x)
        
        # Output
        logits = self.head(x)
        return logits


# ============================================================================
# TRAINING LOOP
# ============================================================================

class Trainer:
    def __init__(self, config, model, train_loader, val_loader, device):
        self.config = config
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        self.optimizer = AdamW(model.parameters(), lr=config.learning_rate)
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=config.num_epochs)
        self.loss_fn = nn.CrossEntropyLoss()
        
        # Setup logging
        os.makedirs(config.log_dir, exist_ok=True)
        self.log_file = os.path.join(config.log_dir, f"training_{datetime.now().isoformat()}.log")
    
    def log(self, message):
        print(message)
        with open(self.log_file, 'a') as f:
            f.write(message + '\n')
    
    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        step = 0
        
        for batch_idx, batch in enumerate(self.train_loader):
            input_ids = batch['input_ids'].to(self.device)
            labels = batch['labels'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            
            # Forward pass
            logits = self.model(input_ids, attention_mask=attention_mask)
            loss = self.loss_fn(logits.view(-1, self.config.vocab_size), labels.view(-1))
            
            # Backward pass with gradient accumulation
            loss = loss / self.config.gradient_accumulation_steps
            loss.backward()
            
            if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()
                step += 1
            
            total_loss += loss.item() * self.config.gradient_accumulation_steps
            
            if batch_idx % self.config.log_every == 0:
                avg_loss = total_loss / (batch_idx + 1)
                self.log(f"Epoch {epoch} | Batch {batch_idx}/{len(self.train_loader)} | Loss: {avg_loss:.4f}")
            
            # Validation checkpoint
            if step % self.config.eval_every == 0 and step > 0:
                val_loss = self.validate()
                self.log(f"Validation Loss: {val_loss:.4f}")
                self.save_checkpoint(epoch, step, val_loss)
        
        self.scheduler.step()
        avg_epoch_loss = total_loss / len(self.train_loader)
        self.log(f"Epoch {epoch} completed | Average Loss: {avg_epoch_loss:.4f}")
        
        return avg_epoch_loss
    
    def validate(self):
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                input_ids = batch['input_ids'].to(self.device)
                labels = batch['labels'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                
                logits = self.model(input_ids, attention_mask=attention_mask)
                loss = self.loss_fn(logits.view(-1, self.config.vocab_size), labels.view(-1))
                total_loss += loss.item()
        
        self.model.train()
        return total_loss / len(self.val_loader)
    
    def save_checkpoint(self, epoch, step, val_loss):
        os.makedirs(self.config.checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(
            self.config.checkpoint_dir,
            f"checkpoint_epoch{epoch}_step{step}_loss{val_loss:.4f}.pt"
        )
        
        torch.save({
            'epoch': epoch,
            'step': step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config.__dict__,
            'val_loss': val_loss
        }, checkpoint_path)
        
        self.log(f"Checkpoint saved: {checkpoint_path}")
    
    def train(self):
        self.log("Starting training...")
        self.log(f"Config: {json.dumps(self.config.__dict__, indent=2, default=str)}")
        
        for epoch in range(self.config.num_epochs):
            self.train_epoch(epoch)
        
        self.log("Training completed!")


# ============================================================================
# MAIN
# ============================================================================

def main():
    config = Config()
    
    # Setup device
    device = torch.device(config.device)
    if config.num_gpus > 1:
        print(f"Using {config.num_gpus} GPUs")
    
    # Create model
    model = TransformerModel(config).to(device)
    if config.num_gpus > 1:
        model = nn.DataParallel(model)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Load data (placeholder tokenizer - replace with actual)
    # from transformers import AutoTokenizer
    # tokenizer = AutoTokenizer.from_pretrained("gpt2")
    
    # For now, use a simple tokenizer placeholder
    class SimpleTokenizer:
        pad_token_id = 0
        def encode(self, text, add_special_tokens=True):
            return [ord(c) % 1000 for c in text][:4096]
    
    tokenizer = SimpleTokenizer()
    
    # Create datasets
    train_dataset = CybersecurityDataset(
        os.path.join(config.data_dir, config.train_file),
        tokenizer,
        max_length=config.max_seq_length
    )
    val_dataset = CybersecurityDataset(
        os.path.join(config.data_dir, config.val_file),
        tokenizer,
        max_length=config.max_seq_length
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=4
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=4
    )
    
    # Train
    trainer = Trainer(config, model, train_loader, val_loader, device)
    trainer.train()


if __name__ == "__main__":
    main()
