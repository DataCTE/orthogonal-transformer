import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
import math
from typing import Optional, Tuple
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import seaborn as sns
from tqdm import tqdm

class OrthogonalTokenEmbedding(nn.Module):
    """
    Creates orthogonal/inverse token embeddings using a learnable orthogonal matrix
    """
    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        
        # Standard embedding
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # Learnable orthogonal transformation matrix
        self.register_buffer('orthogonal_matrix', self._init_orthogonal_matrix(d_model))
        self.transform_weight = nn.Parameter(torch.eye(d_model))
        
    def _init_orthogonal_matrix(self, size: int) -> torch.Tensor:
        """Initialize an orthogonal matrix using QR decomposition"""
        random_matrix = torch.randn(size, size)
        q, _ = torch.linalg.qr(random_matrix)
        return q
    
    def forward(self, x: torch.Tensor, inverse: bool = False) -> torch.Tensor:
        """
        Forward pass with option to get inverse representation
        """
        embed = self.embedding(x)
        
        # Apply learnable transformation
        transform = self.transform_weight @ self.orthogonal_matrix
        
        if inverse:
            # Apply inverse transformation (transpose for orthogonal matrix)
            return embed @ transform.T
        else:
            # Apply forward transformation
            return embed @ transform
    
    def invert_embedding(self, inverse_embed: torch.Tensor) -> torch.Tensor:
        """
        Convert inverse embeddings back to token space
        """
        transform = self.transform_weight @ self.orthogonal_matrix
        standard_embed = inverse_embed @ transform
        
        # Find nearest token in vocabulary
        all_embeddings = self.embedding.weight
        distances = torch.cdist(standard_embed.unsqueeze(1), all_embeddings.unsqueeze(0))
        return distances.argmin(dim=-1)

class MultiHeadOrthogonalAttention(nn.Module):
    """
    Multi-head attention that operates in orthogonal space
    """
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size = query.size(0)
        
        # Linear transformations and reshape
        Q = self.W_q(query).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        
        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        context = torch.matmul(attention_weights, V)
        
        # Reshape and apply output projection
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.W_o(context)
        
        return output

class OrthogonalTransformerBlock(nn.Module):
    """
    Transformer block operating in orthogonal space
    """
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.attention = MultiHeadOrthogonalAttention(d_model, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Self-attention with residual connection
        attn_output = self.attention(x, x, x, mask)
        x = self.norm1(x + attn_output)
        
        # Feed-forward with residual connection
        ff_output = self.feed_forward(x)
        x = self.norm2(x + ff_output)
        
        return x

class OrthogonalNextTokenPredictor(nn.Module):
    """
    Main model that predicts next tokens using orthogonal representations
    """
    def __init__(self, vocab_size: int, d_model: int = 512, n_heads: int = 8,
                 n_layers: int = 6, d_ff: int = 2048, max_seq_len: int = 512,
                 dropout: float = 0.1):
        super().__init__()
        
        self.d_model = d_model
        self.orthogonal_embedding = OrthogonalTokenEmbedding(vocab_size, d_model)
        self.positional_encoding = self._create_positional_encoding(max_seq_len, d_model)
        
        self.transformer_blocks = nn.ModuleList([
            OrthogonalTransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
        # Output projection to inverse token space
        self.output_projection = nn.Linear(d_model, d_model)
        
    def _create_positional_encoding(self, max_seq_len: int, d_model: int) -> torch.Tensor:
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return pe.unsqueeze(0)
    
    def create_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        # Mask should be 1 for positions to attend to (lower triangle including diagonal), 
        # and 0 for positions to ignore (upper triangle).
        # This way, `mask == 0` will be True for future positions, which are then filled with -1e9.
        mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
        return mask
    
    def forward(self, input_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        seq_len = input_ids.size(1)
        device = input_ids.device
        
        # Get inverse embeddings
        x = self.orthogonal_embedding(input_ids, inverse=True)
        
        # Add positional encoding
        x = x + self.positional_encoding[:, :seq_len, :].to(device)
        x = self.dropout(x)
        
        # Create causal mask
        mask = self.create_causal_mask(seq_len, device)
        
        # Pass through transformer blocks
        for block in self.transformer_blocks:
            x = block(x, mask)
        
        x = self.norm(x)
        
        # Project to inverse token prediction space
        inverse_predictions = self.output_projection(x)
        
        # Convert inverse predictions back to token logits
        # This is where we "uninvert" the predictions
        all_inverse_embeddings = self.orthogonal_embedding.embedding.weight @ \
                                (self.orthogonal_embedding.transform_weight @ 
                                 self.orthogonal_embedding.orthogonal_matrix).T
        
        logits = torch.matmul(inverse_predictions, all_inverse_embeddings.T)
        
        return logits, inverse_predictions

class TextDataset(Dataset):
    """Simple text dataset for training"""
    def __init__(self, texts: list, tokenizer, max_length: int = 128):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = []
        
        for text in texts:
            tokens = tokenizer.encode(text)
            if len(tokens) > max_length:
                # Split into chunks
                for i in range(0, len(tokens) - max_length, max_length // 2):
                    self.data.append(tokens[i:i + max_length])
            else:
                self.data.append(tokens)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        tokens = self.data[idx]
        # Pad if necessary
        if len(tokens) < self.max_length:
            tokens = tokens + [0] * (self.max_length - len(tokens))
        return torch.tensor(tokens[:-1]), torch.tensor(tokens[1:])

# Simple character-level tokenizer for demonstration
class CharTokenizer:
    def __init__(self):
        self.char_to_id = {}
        self.id_to_char = {}
        self.vocab_size = 0
    
    def build_vocab(self, texts):
        chars = set()
        for text in texts:
            chars.update(text)
        
        # Add special tokens
        self.char_to_id['<PAD>'] = 0
        self.char_to_id['<UNK>'] = 1
        
        for i, char in enumerate(sorted(chars), 2):
            self.char_to_id[char] = i
        
        self.id_to_char = {v: k for k, v in self.char_to_id.items()}
        self.vocab_size = len(self.char_to_id)
    
    def encode(self, text):
        return [self.char_to_id.get(char, 1) for char in text]
    
    def decode(self, ids):
        return ''.join([self.id_to_char.get(id, '<UNK>') for id in ids])

def train_orthogonal_model(model, train_loader, epochs=10, lr=1e-3, device=None, tokenizer=None, print_every_n_steps=100):
    """Training loop for the orthogonal model"""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model.to(device) # Ensure model is on the correct device

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    pad_token_id = 0 # Default pad token id
    if tokenizer and hasattr(tokenizer, 'pad_id') and tokenizer.pad_id is not None:
        pad_token_id = tokenizer.pad_id
    elif tokenizer and hasattr(tokenizer, 'char_to_id') and '<PAD>' in tokenizer.char_to_id:
        pad_token_id = tokenizer.char_to_id['<PAD>']

    criterion = nn.CrossEntropyLoss(ignore_index=pad_token_id)
    
    model.train()
    for epoch in range(epochs):
        total_epoch_loss = 0
        num_batches = 0
        
        # Variables for step-wise loss reporting
        running_loss_100_steps = 0.0
        steps_in_epoch = 0

        # Progress bar for the epoch
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}", unit="batch")
        
        for batch_idx, (input_ids, target_ids) in enumerate(progress_bar):
            input_ids, target_ids = input_ids.to(device), target_ids.to(device) # Move data to device
            steps_in_epoch += 1
            
            optimizer.zero_grad()
            
            # Forward pass
            logits, inverse_preds = model(input_ids)
            
            # Calculate loss
            loss = criterion(logits.reshape(-1, logits.size(-1)), target_ids.reshape(-1))
            
            # Add orthogonality regularization
            batch_size, seq_len, d_model = inverse_preds.shape
            inverse_flat = inverse_preds.reshape(-1, d_model)
            inverse_norm = F.normalize(inverse_flat, p=2, dim=1)
            corr_matrix = torch.matmul(inverse_norm.T, inverse_norm) / inverse_norm.shape[0]
            identity = torch.eye(d_model, device=device)
            orth_loss = torch.norm(corr_matrix - identity, p='fro')
            
            current_batch_loss = loss + 0.01 * orth_loss
            
            # Backward pass
            current_batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            batch_loss_item = current_batch_loss.item()
            total_epoch_loss += batch_loss_item
            running_loss_100_steps += batch_loss_item
            num_batches +=1
            
            # Update progress bar description with current batch loss and running epoch avg loss
            current_epoch_avg_loss = total_epoch_loss / num_batches
            progress_bar.set_postfix(batch_loss=f"{batch_loss_item:.4f}", epoch_avg_loss=f"{current_epoch_avg_loss:.4f}")

            # Print loss every N steps
            if print_every_n_steps > 0 and steps_in_epoch % print_every_n_steps == 0:
                avg_loss_last_100 = running_loss_100_steps / print_every_n_steps
                print(f"  Epoch {epoch + 1}, Step {steps_in_epoch}/{len(train_loader)}, Avg Loss (last {print_every_n_steps} steps): {avg_loss_last_100:.4f}")
                running_loss_100_steps = 0.0 # Reset for next 100 steps
            
        avg_epoch_loss = total_epoch_loss / num_batches if num_batches > 0 else 0
        # The epoch average loss is already part of the tqdm final line, 
        # but we can print it explicitly if desired, or rely on tqdm's summary.
        # print(f"Orthogonal Model - Epoch {epoch + 1}/{epochs}, Avg Loss: {avg_epoch_loss:.4f}")
        progress_bar.close() # Ensure tqdm progress bar closes properly

def generate_text(model, tokenizer, prompt, max_length=100, temperature=1.0):
    """Generate text using the orthogonal model"""
    model.eval()
    
    # Encode prompt
    input_ids = torch.tensor([tokenizer.encode(prompt)])
    
    # Check if it's a standard model or orthogonal model
    is_orthogonal = hasattr(model, 'orthogonal_embedding')
    
    with torch.no_grad():
        for _ in range(max_length):
            # Get predictions
            if is_orthogonal:
                logits, _ = model(input_ids)
            else:
                logits = model(input_ids)
            
            # Sample from the distribution
            probs = F.softmax(logits[:, -1, :] / temperature, dim=-1)
            next_token = torch.multinomial(probs, 1)
            
            # Append to sequence
            input_ids = torch.cat([input_ids, next_token], dim=1)
            
            # Stop if we hit a special token or max length
            if next_token.item() == 0:  # PAD token
                break
    
    return tokenizer.decode(input_ids[0].tolist())

def analyze_orthogonality(model, tokenizer, sample_text="The quick brown fox"):
    """Analyze the orthogonality of embeddings"""
    model.eval()
    
    # Encode sample text
    input_ids = torch.tensor([tokenizer.encode(sample_text)])
    
    with torch.no_grad():
        # Get standard and inverse embeddings
        standard_embed = model.orthogonal_embedding(input_ids, inverse=False)
        inverse_embed = model.orthogonal_embedding(input_ids, inverse=True)
        
        # Flatten for analysis
        standard_flat = standard_embed.reshape(-1, model.d_model).cpu().numpy()
        inverse_flat = inverse_embed.reshape(-1, model.d_model).cpu().numpy()
        
        # Compute correlation matrices
        standard_corr = np.corrcoef(standard_flat)
        inverse_corr = np.corrcoef(inverse_flat)
        
        # Plot correlation matrices
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        sns.heatmap(standard_corr, ax=ax1, cmap='coolwarm', center=0, vmin=-1, vmax=1)
        ax1.set_title('Standard Embedding Correlations')
        
        sns.heatmap(inverse_corr, ax=ax2, cmap='coolwarm', center=0, vmin=-1, vmax=1)
        ax2.set_title('Inverse/Orthogonal Embedding Correlations')
        
        plt.tight_layout()
        plt.savefig('embedding_correlations.png')
        plt.close()
        
        # Compute orthogonality metrics
        standard_orthogonality = np.mean(np.abs(standard_corr[np.triu_indices_from(standard_corr, k=1)]))
        inverse_orthogonality = np.mean(np.abs(inverse_corr[np.triu_indices_from(inverse_corr, k=1)]))
        
        print(f"\nOrthogonality Analysis:")
        print(f"Standard embedding avg correlation: {standard_orthogonality:.4f}")
        print(f"Inverse embedding avg correlation: {inverse_orthogonality:.4f}")
        print(f"Orthogonality improvement: {(standard_orthogonality - inverse_orthogonality) / standard_orthogonality * 100:.2f}%")
        
        # PCA visualization
        pca = PCA(n_components=2)
        standard_pca = pca.fit_transform(standard_flat)
        inverse_pca = pca.fit_transform(inverse_flat)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Plot with labels
        tokens = tokenizer.decode(input_ids[0].tolist()).replace('<PAD>', '').replace('<UNK>', '')
        for i, char in enumerate(tokens[:len(standard_pca)]):
            ax1.scatter(standard_pca[i, 0], standard_pca[i, 1])
            ax1.annotate(char, (standard_pca[i, 0], standard_pca[i, 1]))
            
            ax2.scatter(inverse_pca[i, 0], inverse_pca[i, 1])
            ax2.annotate(char, (inverse_pca[i, 0], inverse_pca[i, 1]))
        
        ax1.set_title('Standard Embeddings (PCA)')
        ax2.set_title('Inverse/Orthogonal Embeddings (PCA)')
        
        plt.tight_layout()
        plt.savefig('embedding_pca.png')
        plt.close()

def compare_predictions(model, tokenizer, prompts):
    """Compare predictions in standard vs inverse space"""
    model.eval()
    
    results = []
    for prompt in prompts:
        input_ids = torch.tensor([tokenizer.encode(prompt)])
        
        with torch.no_grad():
            # Get predictions
            logits, inverse_preds = model(input_ids)
            
            # Get top 5 predictions
            top5_probs, top5_tokens = torch.topk(F.softmax(logits[0, -1], dim=-1), k=5)
            
            result = {
                'prompt': prompt,
                'predictions': [(tokenizer.decode([tok.item()]), prob.item()) 
                               for tok, prob in zip(top5_tokens, top5_probs)]
            }
            results.append(result)
    
    print("\nPrediction Analysis:")
    for result in results:
        print(f"\nPrompt: '{result['prompt']}'")
        print("Top 5 predictions:")
        for i, (char, prob) in enumerate(result['predictions']):
            print(f"  {i+1}. '{char}' ({prob:.3f})")

class StandardTransformer(nn.Module):
    """Standard transformer for comparison"""
    def __init__(self, vocab_size: int, d_model: int = 512, n_heads: int = 8,
                 n_layers: int = 6, d_ff: int = 2048, max_seq_len: int = 512,
                 dropout: float = 0.1):
        super().__init__()
        
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = OrthogonalNextTokenPredictor._create_positional_encoding(
            None, max_seq_len, d_model
        )
        
        self.transformer_blocks = nn.ModuleList([
            OrthogonalTransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.output_projection = nn.Linear(d_model, vocab_size)
        
    def create_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        # Mask should be 1 for positions to attend to (lower triangle including diagonal), 
        # and 0 for positions to ignore (upper triangle).
        # This way, `mask == 0` will be True for future positions, which are then filled with -1e9.
        mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
        return mask
        
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        seq_len = input_ids.size(1)
        device = input_ids.device
        
        # Standard embeddings
        x = self.embedding(input_ids) * math.sqrt(self.d_model)
        x = x + self.positional_encoding[:, :seq_len, :].to(device)
        x = self.dropout(x)
        
        # Create causal mask
        mask = self.create_causal_mask(seq_len, device)
        
        # Pass through transformer blocks
        for block in self.transformer_blocks:
            x = block(x, mask)
        
        x = self.norm(x)
        logits = self.output_projection(x)
        
        return logits

def train_and_compare():
    """Train both standard and orthogonal models for comparison"""
    # Expanded training data
    texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning models can learn patterns from data.",
        "Orthogonal representations may reduce bias in predictions.",
        "This is a test of the inverse token prediction system.",
        "Neural networks can approximate complex functions.",
        "Deep learning has revolutionized artificial intelligence.",
        "Transformers are powerful sequence modeling architectures.",
        "Attention mechanisms allow models to focus on relevant information.",
        "The cat sat on the mat and looked out the window.",
        "Programming is the art of telling computers what to do.",
        "Mathematics is the language of the universe.",
        "Science discovers the laws that govern our world.",
        "Technology transforms how we live and work.",
        "Innovation drives progress in modern society.",
        "Education empowers individuals to reach their potential."
    ]
    
    # Initialize tokenizer
    tokenizer = CharTokenizer()
    tokenizer.build_vocab(texts)
    
    # Create dataset
    dataset = TextDataset(texts, tokenizer, max_length=50)
    train_loader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    # Initialize both models with same architecture
    model_params = dict(
        vocab_size=tokenizer.vocab_size,
        d_model=128,
        n_heads=4,
        n_layers=3,
        d_ff=512,
        max_seq_len=100
    )
    
    orthogonal_model = OrthogonalNextTokenPredictor(**model_params)
    standard_model = StandardTransformer(**model_params)
    
    print("Training Orthogonal Model...")
    train_orthogonal_model(orthogonal_model, train_loader, epochs=10)
    
    print("\nTraining Standard Model...")
    optimizer = torch.optim.Adam(standard_model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    
    standard_model.train()
    for epoch in range(10):
        total_loss = 0
        for input_ids, target_ids in train_loader:
            optimizer.zero_grad()
            logits = standard_model(input_ids)
            loss = criterion(logits.reshape(-1, logits.size(-1)), target_ids.reshape(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}/10, Loss: {total_loss / len(train_loader):.4f}")
    
    # Compare generations
    print("\n" + "="*50)
    print("GENERATION COMPARISON")
    print("="*50)
    
    test_prompts = ["The ", "Machine ", "Deep ", "Science "]
    
    for prompt in test_prompts:
        print(f"\nPrompt: '{prompt}'")
        
        orthogonal_gen = generate_text(orthogonal_model, tokenizer, prompt, max_length=30, temperature=0.8)
        standard_gen = generate_text(standard_model, tokenizer, prompt, max_length=30, temperature=0.8)
        
        print(f"Orthogonal: {orthogonal_gen}")
        print(f"Standard:   {standard_gen}")
    
    return orthogonal_model, standard_model, tokenizer

if __name__ == "__main__":
    # Example usage
    print("Orthogonal Next Token Predictor Demo")
    print("=" * 50)
    
    # Train and compare models
    orthogonal_model, standard_model, tokenizer = train_and_compare()
    
    # Analyze orthogonality
    print("\n" + "="*50)
    print("ORTHOGONALITY ANALYSIS")
    print("="*50)
    analyze_orthogonality(orthogonal_model, tokenizer)
    
    # Compare prediction distributions
    print("\n" + "="*50)
    print("PREDICTION DISTRIBUTION ANALYSIS")
    print("="*50)
    compare_predictions(orthogonal_model, tokenizer, 
                       ["The ", "Machine ", "Neural ", "Transform"])
    
    print("\nAnalysis complete! Check 'embedding_correlations.png' and 'embedding_pca.png' for visualizations.")
