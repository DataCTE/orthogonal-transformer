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

class OrthogonalProjectedTokenEmbedding(nn.Module):
    """
    Creates double orthogonal embeddings: first transforms to orthogonal space,
    then projects to a second orthogonal space (both in d_model dimensions).
    """
    def __init__(self, vocab_size: int, d_model: int, d_projection: int = None):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        # d_projection is kept for compatibility but should equal d_model
        self.d_projection = d_model if d_projection is None else d_projection
        if self.d_projection != self.d_model:
            print(f"Warning: d_projection ({self.d_projection}) != d_model ({self.d_model}). Using d_model for orthogonal projection.")
            self.d_projection = self.d_model
        
        # Standard embedding (same as baseline)
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # First orthogonal transformation (same as baseline)
        self.register_buffer('orthogonal_matrix', self._init_orthogonal_matrix(d_model))
        self.transform_weight = nn.Parameter(torch.eye(d_model))
        
        # Second orthogonal transformation for projection space
        self.register_buffer('projection_orthogonal_matrix', self._init_orthogonal_matrix(d_model))
        self.projection_transform_weight = nn.Parameter(torch.eye(d_model))
        
    def _init_orthogonal_matrix(self, size: int) -> torch.Tensor:
        """Initialize an orthogonal matrix using QR decomposition"""
        random_matrix = torch.randn(size, size)
        q, _ = torch.linalg.qr(random_matrix)
        return q
    
    def forward(self, x: torch.Tensor, inverse: bool = False, project: bool = True) -> torch.Tensor:
        """
        Forward pass with double orthogonal transformation.
        inverse: Apply inverse transformation (baseline behavior)
        project: Apply second orthogonal transformation
        """
        embed = self.embedding(x)
        
        # First orthogonal transformation (same as baseline)
        transform = self.transform_weight @ self.orthogonal_matrix
        
        if inverse:
            orthogonal_embed = embed @ transform.T
        else:
            orthogonal_embed = embed @ transform
        
        # Second orthogonal transformation (the "projection")
        if project:
            projection_transform = self.projection_transform_weight @ self.projection_orthogonal_matrix
            # This is still d_model -> d_model, but in a different orthogonal basis
            projected = orthogonal_embed @ projection_transform
            return projected
        else:
            return orthogonal_embed
    
    def invert_embedding(self, embedding: torch.Tensor, from_projected: bool = True) -> torch.Tensor:
        """
        Convert embeddings back to token space.
        """
        transform = self.transform_weight @ self.orthogonal_matrix
        
        if from_projected:
            # First undo the projection transformation
            projection_transform = self.projection_transform_weight @ self.projection_orthogonal_matrix
            # Inverse of orthogonal matrix is its transpose
            unprojected = embedding @ projection_transform.T
            # Then undo the base transformation
            standard_embed = unprojected @ transform
        else:
            # Just undo the base transformation
            standard_embed = embedding @ transform
            
        # Find nearest token
        all_embeddings = self.embedding.weight
        distances = torch.cdist(standard_embed.unsqueeze(1), all_embeddings.unsqueeze(0))
        return distances.argmin(dim=-1)

class MultiHeadOrthogonalProjectedAttention(nn.Module):
    """
    Multi-head attention operating in orthogonal space.
    """
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        # Same structure as baseline
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size = query.size(0)
        
        Q = self.W_q(query).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        context = torch.matmul(attention_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.W_o(context)
        
        return output

class OrthogonalProjectedTransformerBlock(nn.Module):
    """
    Transformer block operating in orthogonal projected space.
    """
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.attention = MultiHeadOrthogonalProjectedAttention(d_model, n_heads, dropout)
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
        attn_output = self.attention(x, x, x, mask)
        x = self.norm1(x + attn_output)
        
        ff_output = self.feed_forward(x)
        x = self.norm2(x + ff_output)
        
        return x

class ProjectedSpaceTransformer(nn.Module):
    """
    Transformer that operates in a double orthogonal space.
    First applies baseline orthogonal transformation, then projects to another orthogonal space.
    """
    def __init__(self, vocab_size: int, d_model: int = 512, d_projection: int = None,
                 n_heads: int = 8, n_layers: int = 6, d_ff: int = 2048, 
                 d_ff_projection: int = None, max_seq_len: int = 512,
                 dropout: float = 0.1):
        super().__init__()
        
        self.d_model = d_model
        # Force d_projection to equal d_model for true orthogonal projection
        self.d_projection = d_model
        self.use_projection = True  # Always use double orthogonal transformation
        
        # Use extended embedding layer
        self.orthogonal_embedding = OrthogonalProjectedTokenEmbedding(vocab_size, d_model, self.d_projection)
        
        # Positional encoding in d_model space
        self.positional_encoding = self._create_positional_encoding(max_seq_len, d_model)
        
        # Transformer blocks in d_model space
        d_ff_working = d_ff if d_ff_projection is None else d_ff
        self.transformer_blocks = nn.ModuleList([
            OrthogonalProjectedTransformerBlock(d_model, n_heads, d_ff_working, dropout)
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
        mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
        return mask
    
    def forward(self, input_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        seq_len = input_ids.size(1)
        device = input_ids.device
        
        # Get double orthogonal embeddings (inverse + projected)
        x = self.orthogonal_embedding(input_ids, inverse=True, project=True)
        
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
        # Get all vocabulary embeddings in double orthogonal space
        all_embeddings = self.orthogonal_embedding.embedding.weight
        
        # Apply first orthogonal transformation
        transform = self.orthogonal_embedding.transform_weight @ self.orthogonal_embedding.orthogonal_matrix
        all_inverse = all_embeddings @ transform.T
        
        # Apply second orthogonal transformation
        projection_transform = self.orthogonal_embedding.projection_transform_weight @ self.orthogonal_embedding.projection_orthogonal_matrix
        all_projected = all_inverse @ projection_transform
        
        logits = torch.matmul(inverse_predictions, all_projected.T)
        
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

def train_projected_space_model(model, train_loader, epochs=10, lr=1e-3, device=None, tokenizer=None, print_every_n_steps=100):
    """Training loop - follows baseline train_orthogonal_model exactly"""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    pad_token_id = 0
    if tokenizer and hasattr(tokenizer, 'pad_id') and tokenizer.pad_id is not None:
        pad_token_id = tokenizer.pad_id
    elif tokenizer and hasattr(tokenizer, 'char_to_id') and '<PAD>' in tokenizer.char_to_id:
        pad_token_id = tokenizer.char_to_id['<PAD>']

    criterion = nn.CrossEntropyLoss(ignore_index=pad_token_id)
    
    model.train()
    for epoch in range(epochs):
        total_epoch_loss = 0
        num_batches = 0
        
        running_loss_100_steps = 0.0
        steps_in_epoch = 0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}", unit="batch")
        
        for batch_idx, (input_ids, target_ids) in enumerate(progress_bar):
            input_ids, target_ids = input_ids.to(device), target_ids.to(device)
            steps_in_epoch += 1
            
            optimizer.zero_grad()
            
            logits, inverse_preds = model(input_ids)
            
            loss = criterion(logits.reshape(-1, logits.size(-1)), target_ids.reshape(-1))
            
            # Add orthogonality regularization (same as baseline)
            working_dim = inverse_preds.shape[-1]
            batch_size, seq_len, _ = inverse_preds.shape
            inverse_flat = inverse_preds.reshape(-1, working_dim)
            inverse_norm = F.normalize(inverse_flat, p=2, dim=1)
            corr_matrix = torch.matmul(inverse_norm.T, inverse_norm) / inverse_norm.shape[0]
            identity = torch.eye(working_dim, device=device)
            orth_loss = torch.norm(corr_matrix - identity, p='fro')
            
            current_batch_loss = loss + 0.01 * orth_loss
            
            current_batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            batch_loss_item = current_batch_loss.item()
            total_epoch_loss += batch_loss_item
            running_loss_100_steps += batch_loss_item
            num_batches += 1
            
            current_epoch_avg_loss = total_epoch_loss / num_batches
            progress_bar.set_postfix(batch_loss=f"{batch_loss_item:.4f}", epoch_avg_loss=f"{current_epoch_avg_loss:.4f}")

            if print_every_n_steps > 0 and steps_in_epoch % print_every_n_steps == 0:
                avg_loss_last_100 = running_loss_100_steps / print_every_n_steps
                print(f"  Epoch {epoch + 1}, Step {steps_in_epoch}/{len(train_loader)}, Avg Loss (last {print_every_n_steps} steps): {avg_loss_last_100:.4f}")
                running_loss_100_steps = 0.0
            
        avg_epoch_loss = total_epoch_loss / num_batches if num_batches > 0 else 0
        progress_bar.close()

def generate_text(model, tokenizer, prompt, max_length=100, temperature=1.0):
    """Generate text using the model"""
    model.eval()
    device = next(model.parameters()).device
    input_ids = torch.tensor([tokenizer.encode(prompt)], device=device)
    
    is_projected_space_model = hasattr(model, 'orthogonal_embedding')

    with torch.no_grad():
        for _ in range(max_length):
            if is_projected_space_model:
                logits, _ = model(input_ids)
            elif isinstance(model, StandardTransformer):
                logits = model(input_ids)
            else:
                print(f"Model type {type(model)} not recognized for generate_text in this script.")
                return None
            
            probs = F.softmax(logits[:, -1, :] / temperature, dim=-1)
            next_token = torch.multinomial(probs, 1)
            
            input_ids = torch.cat([input_ids, next_token], dim=1)
            
            pad_token_id = tokenizer.char_to_id.get('<PAD>', 0)
            if next_token.item() == pad_token_id:
                break
    
    return tokenizer.decode(input_ids[0].tolist())

def analyze_orthogonality(model, tokenizer, sample_text="The quick brown fox"):
    """Analyze the orthogonality of embeddings"""
    model.eval()
    
    input_ids = torch.tensor([tokenizer.encode(sample_text)]).to(next(model.parameters()).device)
    
    with torch.no_grad():
        if hasattr(model, 'orthogonal_embedding'):
            # Get embeddings without projection
            standard_embed = model.orthogonal_embedding(input_ids, inverse=False, project=False)
            # Get embeddings with double orthogonal transformation
            projected_embed = model.orthogonal_embedding(input_ids, inverse=True, project=True)
            
            # Visualize the transformation matrices
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
            
            # First orthogonal transformation
            transform1 = model.orthogonal_embedding.transform_weight @ model.orthogonal_embedding.orthogonal_matrix
            im1 = ax1.imshow(transform1.cpu().numpy(), aspect='auto', cmap='coolwarm')
            ax1.set_title('First Orthogonal Transformation Matrix')
            ax1.set_xlabel('Output Dimensions')
            ax1.set_ylabel('Input Dimensions')
            plt.colorbar(im1, ax=ax1)
            
            # Second orthogonal transformation
            transform2 = model.orthogonal_embedding.projection_transform_weight @ model.orthogonal_embedding.projection_orthogonal_matrix
            im2 = ax2.imshow(transform2.cpu().numpy(), aspect='auto', cmap='coolwarm')
            ax2.set_title('Second Orthogonal Transformation Matrix')
            ax2.set_xlabel('Output Dimensions')
            ax2.set_ylabel('Input Dimensions')
            plt.colorbar(im2, ax=ax2)
            
            plt.tight_layout()
            plt.savefig('orthogonal_transformations.png')
            plt.close()
            print(f"Orthogonal transformation matrices saved to 'orthogonal_transformations.png'")
        else:
            print("Model does not have recognizable embedding structure for analysis.")
            return

        standard_flat = standard_embed.reshape(-1, model.d_model).cpu().numpy()
        current_projection_dim = projected_embed.shape[-1]
        projected_flat = projected_embed.reshape(-1, current_projection_dim).cpu().numpy()
        
        if standard_flat.shape[0] <= 1 or projected_flat.shape[0] <= 1:
            print("Not enough tokens in sample text to compute correlation.")
            return

        standard_corr = np.corrcoef(standard_flat.T)
        projected_corr = np.corrcoef(projected_flat.T)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        sns.heatmap(standard_corr, ax=ax1, cmap='coolwarm', center=0, vmin=-1, vmax=1)
        ax1.set_title('Standard Embedding Dimension Correlations')
        
        sns.heatmap(projected_corr, ax=ax2, cmap='coolwarm', center=0, vmin=-1, vmax=1)
        ax2.set_title('Double Orthogonal Space Dimension Correlations')
        
        plt.tight_layout()
        plt.savefig('embedding_correlations.png')
        plt.close()
        
        standard_orthogonality = np.mean(np.abs(standard_corr[np.triu_indices_from(standard_corr, k=1)]))
        projected_orthogonality = np.mean(np.abs(projected_corr[np.triu_indices_from(projected_corr, k=1)]))
        
        print(f"\nDimensionality Correlation Analysis (Lower is better for disentanglement):")
        print(f"Standard (d_model={model.d_model}) embedding avg off-diagonal correlation: {standard_orthogonality:.4f}")
        print(f"Projected (d_projection={current_projection_dim}) space avg off-diagonal correlation: {projected_orthogonality:.4f}")
        
        if standard_flat.shape[0] < 2 or projected_flat.shape[0] < 2 :
            print("Not enough tokens for PCA visualization.")
        else:
            pca_std = PCA(n_components=min(2, standard_flat.shape[0], standard_flat.shape[1]))
            standard_pca = pca_std.fit_transform(standard_flat)
            
            pca_projected = PCA(n_components=min(2, projected_flat.shape[0], projected_flat.shape[1]))
            projected_pca = pca_projected.fit_transform(projected_flat)
            
            fig_pca, (ax_pca1, ax_pca2) = plt.subplots(1, 2, figsize=(12, 5))
            
            tokens = tokenizer.decode(input_ids[0].tolist()).replace('<PAD>', '').replace('<UNK>', '')
            num_plot_tokens = min(len(tokens), standard_pca.shape[0], projected_pca.shape[0])

            for i in range(num_plot_tokens):
                ax_pca1.scatter(standard_pca[i, 0], standard_pca[i, 1] if standard_pca.shape[1]>1 else 0)
                ax_pca1.annotate(tokens[i], (standard_pca[i, 0], standard_pca[i, 1] if standard_pca.shape[1]>1 else 0))
                
                ax_pca2.scatter(projected_pca[i, 0], projected_pca[i, 1] if projected_pca.shape[1]>1 else 0)
                ax_pca2.annotate(tokens[i], (projected_pca[i, 0], projected_pca[i, 1] if projected_pca.shape[1]>1 else 0))
            
            ax_pca1.set_title('Standard Embeddings (PCA)')
            ax_pca2.set_title('Projected Space Embeddings (PCA)')
            
            plt.tight_layout()
            plt.savefig('embedding_pca.png')
            plt.close()

def compare_predictions(model, tokenizer, prompts):
    """Compare predictions in standard vs projected space"""
    model.eval()
    
    results = []
    for prompt in prompts:
        input_ids = torch.tensor([tokenizer.encode(prompt)])
        
        with torch.no_grad():
            if hasattr(model, 'token_embedder'):
                logits, _ = model(input_ids)
            else:
                print("Model structure not recognized for compare_predictions")
                return
            
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

class StandardTransformerBlock(nn.Module):
    """
    Standard Transformer block operating in d_model space.
    This is what OrthogonalTransformerBlock used to be.
    """
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        # Attention should be a standard multi-head attention operating on d_model
        # MultiHeadProjectedAttention is for d_projection. We need a MultiHeadAttention for d_model.
        # Let's assume MultiHeadProjectedAttention can be reused if d_model is passed as d_projection,
        # or define a separate MultiHeadStandardAttention.
        # For now, let's define a new MultiHeadStandardAttention for clarity.
        self.attention = MultiHeadStandardAttention(d_model, n_heads, dropout)
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
        attn_output = self.attention(x, x, x, mask)
        x = self.norm1(x + attn_output)
        ff_output = self.feed_forward(x)
        x = self.norm2(x + ff_output)
        return x

class MultiHeadStandardAttention(nn.Module):
    """
    Standard Multi-head attention that operates in d_model space.
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
        
        Q = self.W_q(query).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        context = torch.matmul(attention_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.W_o(context)
        return output

class StandardTransformer(nn.Module):
    """Standard transformer for comparison"""
    def __init__(self, vocab_size: int, d_model: int = 512, n_heads: int = 8,
                 n_layers: int = 6, d_ff: int = 2048, max_seq_len: int = 512,
                 dropout: float = 0.1):
        super().__init__()
        
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        pe_temp = torch.zeros(max_seq_len, d_model)
        position_temp = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term_temp = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe_temp[:, 0::2] = torch.sin(position_temp * div_term_temp)
        pe_temp[:, 1::2] = torch.cos(position_temp * div_term_temp)
        self.register_buffer('positional_encoding', pe_temp.unsqueeze(0))

        self.transformer_blocks = nn.ModuleList([
            StandardTransformerBlock(d_model, n_heads, d_ff, dropout)
        for _ in range(n_layers)
        ])
        
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.output_projection = nn.Linear(d_model, vocab_size)
        
    def create_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
        return mask
        
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        seq_len = input_ids.size(1)
        device = input_ids.device
        
        x = self.embedding(input_ids) * math.sqrt(self.d_model)
        x = x + self.positional_encoding[:, :seq_len, :].to(device)
        x = self.dropout(x)
        
        mask = self.create_causal_mask(seq_len, device)
        
        for block in self.transformer_blocks:
            x = block(x, mask)
        
        x = self.norm(x)
        logits = self.output_projection(x)
        
        return logits

def train_and_compare():
    """Train both standard and orthogonal models for comparison"""
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
    
    tokenizer = CharTokenizer()
    tokenizer.build_vocab(texts)
    
    dataset = TextDataset(texts, tokenizer, max_length=50)
    train_loader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    d_model_val = 128
    d_projection_val = 64  # Actually use half the dimension!
    d_ff_model = 512
    d_ff_projection_val = 256

    model_params_projected = dict(
        vocab_size=tokenizer.vocab_size,
        d_model=d_model_val,
        d_projection=d_projection_val,
        n_heads=4,
        n_layers=3,
        d_ff_projection=d_ff_projection_val,
        max_seq_len=100,
        dropout=0.1
    )
    
    model_params_std = dict(
        vocab_size=tokenizer.vocab_size,
        d_model=d_model_val,
        n_heads=4,
        n_layers=3,
        d_ff=d_ff_model,
        max_seq_len=100,
        dropout=0.1
    )
    
    projected_space_model = ProjectedSpaceTransformer(**model_params_projected)
    standard_model = StandardTransformer(**model_params_std)
    
    print("Training ProjectedSpace Model...")
    train_projected_space_model(projected_space_model, train_loader, epochs=10, tokenizer=tokenizer, print_every_n_steps=100)
    
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
    
    print("\n" + "="*50)
    print("GENERATION COMPARISON")
    print("="*50)
    
    test_prompts = ["The ", "Machine ", "Deep ", "Science "]
    
    for prompt in test_prompts:
        print(f"\nPrompt: '{prompt}'")
        
        projected_space_gen = generate_text(projected_space_model, tokenizer, prompt, max_length=30, temperature=0.8)
        standard_gen = generate_text(standard_model, tokenizer, prompt, max_length=30, temperature=0.8)
        
        print(f"ProjectedSpace: {projected_space_gen}")
        print(f"Standard:   {standard_gen}")
    
    return projected_space_model, standard_model, tokenizer

if __name__ == "__main__":
    print("ProjectedSpace Transformer Demo")
    print("=" * 50)
    
    projected_space_model, standard_model, tokenizer = train_and_compare()
    
    print("\n" + "="*50)
    print("PROJECTED SPACE ANALYSIS")
    print("="*50)
    analyze_orthogonality(projected_space_model, tokenizer)
    
    print("\n" + "="*50)
    print("PREDICTION DISTRIBUTION ANALYSIS")
    print("="*50)
    compare_predictions(projected_space_model, tokenizer, 
                       ["The ", "Machine ", "Neural ", "Transform"])
    
    print("\nAnalysis complete! Check 'embedding_correlations.png' and 'embedding_pca.png' for visualizations.")
