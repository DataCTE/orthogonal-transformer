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

class ProjectedTokenEmbedding(nn.Module):
    """
    Creates token embeddings and their linearly projected representations.
    """
    def __init__(self, vocab_size: int, d_model: int, d_projection: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.d_projection = d_projection 

        if self.d_projection != self.d_model:
             print(f"Warning: ProjectedTokenEmbedding often expects d_projection ({d_projection}) == d_model ({d_model}) for these experiments. Ensure this is intended if they differ.")

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.projection_layer = nn.Linear(d_model, self.d_projection)
        
    def _init_orthogonal_matrix(self, size: int) -> torch.Tensor: # This method is no longer used by this class
        # Kept for reference or if switching back, but not active for current nn.Linear approach
        random_matrix = torch.randn(size, size)
        q, _ = torch.linalg.qr(random_matrix)
        return q

    def forward(self, x: torch.Tensor, to_projected_space: bool = False) -> torch.Tensor:
        """
        Forward pass.
        If to_projected_space is True, returns the linearly projected (projected) representation.
        Otherwise (for tests like embedding independence), returns the standard d_model embedding.
        """
        embed = self.embedding(x) # (batch, seq_len, d_model)
        
        if to_projected_space:
            # Transform to the "projected" space used for processing
            return self.projection_layer(embed)
        else:
            # Return the original standard embedding if not transforming to projected space
            # This is useful for comparing against the projected space in tests.
            return embed

    def get_logits_from_projected_space(self, projected_output: torch.Tensor) -> torch.Tensor:
        """
        Calculates logits from a projected space representation.
        projected_output: tensor from the transformer stack, shape (batch, seq_len, d_projection) 
                       (where d_projection == d_model in the current setup)
        """
        # Project all vocabulary embeddings into the same "projected" space
        all_vocab_standard_embeds = self.embedding.weight # (vocab_size, d_model)
        all_vocab_projected_embeds = self.projection_layer(all_vocab_standard_embeds) # (vocab_size, d_projection)

        # Ensure projected_output is at least 2D for matmul: (N, d_projection)
        original_shape = projected_output.shape
        if projected_output.ndim == 1: 
            projected_output = projected_output.unsqueeze(0)
        elif projected_output.ndim > 2: 
            projected_output = projected_output.reshape(-1, self.d_projection)
        
        # Compute dot product similarity
        logits = torch.matmul(projected_output, all_vocab_projected_embeds.T)
        
        if len(original_shape) > 2:
            logits = logits.reshape(original_shape[0], original_shape[1], -1)
        elif len(original_shape) == 1 and logits.shape[0] == 1:
            logits = logits.squeeze(0)

        return logits

class MultiHeadProjectedAttention(nn.Module):
    """
    Multi-head attention that operates in the projected space (d_projection)
    """
    def __init__(self, d_projection: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_projection % n_heads == 0
        
        self.d_projection = d_projection
        self.n_heads = n_heads
        self.d_k = d_projection // n_heads
        
        self.W_q = nn.Linear(d_projection, d_projection)
        self.W_k = nn.Linear(d_projection, d_projection)
        self.W_v = nn.Linear(d_projection, d_projection)
        self.W_o = nn.Linear(d_projection, d_projection)
        
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
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_projection)
        output = self.W_o(context)
        
        return output

class ProjectedSpaceTransformerBlock(nn.Module):
    """
    Transformer block operating in projected space (d_projection)
    """
    def __init__(self, d_projection: int, n_heads: int, d_ff_projection: int, dropout: float = 0.1):
        super().__init__()
        self.attention = MultiHeadProjectedAttention(d_projection, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_projection)
        self.norm2 = nn.LayerNorm(d_projection)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(d_projection, d_ff_projection),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff_projection, d_projection),
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
    Predicts next tokens by reasoning in a transformed "projected space".
    """
    def __init__(self, vocab_size: int, d_model: int, d_projection: int,
                 n_heads: int = 8, n_layers: int = 6, d_ff_projection: int = 2048,
                 max_seq_len: int = 512, dropout: float = 0.1):
        super().__init__()
        
        self.d_model = d_model
        self.d_projection = d_projection
        
        self.token_embedder = ProjectedTokenEmbedding(vocab_size, d_model, d_projection)
        
        self.positional_encoding = self._create_positional_encoding(max_seq_len, self.d_projection)
        
        self.transformer_blocks = nn.ModuleList([
            ProjectedSpaceTransformerBlock(d_projection, n_heads, d_ff_projection, dropout)
            for _ in range(n_layers)
        ])
        
        self.norm = nn.LayerNorm(d_projection)
        self.dropout = nn.Dropout(dropout)
        
    def _create_positional_encoding(self, max_seq_len: int, dim: int) -> torch.Tensor:
        pe = torch.zeros(max_seq_len, dim)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * 
                           (-math.log(10000.0) / dim))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return pe.unsqueeze(0)
    
    def create_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
        return mask
    
    def forward(self, input_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        seq_len = input_ids.size(1)
        device = input_ids.device
        
        x = self.token_embedder(input_ids, to_projected_space=True)
        
        x = x + self.positional_encoding[:, :seq_len, :].to(device)
        x = self.dropout(x)
        
        mask = self.create_causal_mask(seq_len, device)
        
        for block in self.transformer_blocks:
            x = block(x, mask)
        
        projected_reprs = self.norm(x)
        
        logits = self.token_embedder.get_logits_from_projected_space(projected_reprs)
        
        return logits, projected_reprs

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

def train_orthogonal_model(model, train_loader, epochs=10, lr=1e-3, device=None, tokenizer=None, 
                           print_every_n_steps=100, orth_loss_weight: float = 0.01,
                           aux_relational_loss_weight: float = 0.001): # New auxiliary loss weight
    """Training loop for the ProjectedSpace model, with projected space regularization and auxiliary relational loss."""
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
    aux_criterion = nn.MSELoss() # For predicting cosine similarities

    model.train()
    for epoch in range(epochs):
        total_epoch_loss = 0
        num_batches = 0
        
        running_loss_100_steps = 0.0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}", unit="batch")
        
        for batch_idx, (input_ids_full, target_ids) in enumerate(progress_bar): # input_ids_full includes the last token for target
            # We need original input_ids for the auxiliary task, not just input_ids for model fwd pass
            # input_ids for model: input_ids_full[:, :-1]
            # target_ids for model: input_ids_full[:, 1:] 
            # (Assuming TextDataset returns full sequence and then slices. Let's adjust if needed)
            # For simplicity, let's assume train_loader gives (input_ids_for_model, target_ids_for_model)
            # and we can reconstruct the original token sequence that led to input_ids_for_model for aux task
            
            input_ids, target_ids = input_ids_full.to(device), target_ids.to(device) # Assuming train_loader now gives (input_ids, target_ids) suitable for model
            
            optimizer.zero_grad()
            
            logits, projected_reprs = model(input_ids) # projected_reprs are (batch, seq_len, d_projection)
            
            main_loss = criterion(logits.reshape(-1, logits.size(-1)), target_ids.reshape(-1))
            loss = main_loss
            
            # Orthogonality regularization on projected_reprs
            if orth_loss_weight > 0 and hasattr(model, 'd_projection'):
                batch_size_orth, seq_len_orth, d_projection_dim_orth = projected_reprs.shape
                projected_flat = projected_reprs.reshape(-1, d_projection_dim_orth)
                projected_norm = F.normalize(projected_flat, p=2, dim=1)
                if projected_norm.shape[0] > 1 and projected_norm.shape[1] > 1 :
                    corr_matrix = torch.matmul(projected_norm.T, projected_norm) / projected_norm.shape[0]
                    identity = torch.eye(d_projection_dim_orth, device=device)
                    orth_loss_val = torch.norm(corr_matrix - identity, p='fro')
                    loss = loss + orth_loss_weight * orth_loss_val

            # Auxiliary Relational Loss
            if aux_relational_loss_weight > 0 and input_ids.size(0) > 1 and input_ids.size(1) > 1: # Need at least 2 items in batch and seq for pairs
                # Get initial projected embeddings (without positional, pre-transformer)
                # This requires access to the raw token_ids that formed the input_ids sequence
                # For simplicity, let's use the input_ids themselves (which are token IDs)
                # and project them through the embedder's projection_layer
                
                # We need the *original* full sequence of embeddings for comparison.
                # model.token_embedder.embedding(input_ids) gives (batch, seq_len, d_model)
                # model.token_embedder.projection_layer(...) projects these.

                # Let's try to predict similarity of initial *projected* embeddings from *contextualized* projected_reprs
                # This is a bit indirect. A more direct way would be to compare projected embeddings of tokens
                # from the original input sequence.
                
                # Simplified approach: try to reconstruct pairwise cosine similarities
                # of the *initial projected embeddings* from the *final projected_reprs*.
                # This is complex to set up efficiently in the loop.

                # Alternative simpler auxiliary task:
                # Can the model predict the distance between its own projected_reprs for t and t+1?
                # This doesn't directly involve "seeing all representations" but internal consistency.

                # Let's stick to a simpler concept for now that is easier to implement:
                # Ensure the `projected_reprs` (final contextualized representations) for adjacent tokens
                # are not *too* dissimilar if the original tokens were similar, or *too* similar if different.
                # This still requires a notion of original token similarity.

                # --- Simplest Auxiliary: Predict variance of projected_reprs along sequence ----
                # This encourages projected_reprs to change, promoting some dynamic. Low weight.
                # This is not directly "seeing distances between all representations"
                # but a very gentle structural nudge.
                
                # For a more direct "seeing distances":
                # We need to compare pairs. Let's take first two elements of seq in projected_reprs
                # And compare with first two elements of initial projected embeddings
                
                if input_ids.size(1) >= 2: # Need at least two tokens in sequence
                    # Get initial projected embeddings for the first two tokens of each batch item
                    # We assume input_ids are the token indices for the model's input sequence
                    initial_embeds = model.token_embedder.embedding(input_ids[:, :2]) # (batch, 2, d_model)
                    projected_initial_embeds = model.token_embedder.projection_layer(initial_embeds) # (batch, 2, d_projection)
                    
                    # Calculate target cosine similarity for these initial projected pairs
                    # projected_initial_embeds[:, 0, :] is (batch, d_projection)
                    # projected_initial_embeds[:, 1, :] is (batch, d_projection)
                    target_sim = F.cosine_similarity(projected_initial_embeds[:, 0, :], projected_initial_embeds[:, 1, :], dim=1) # (batch)
                    
                    # Predict this similarity from the corresponding projected_reprs
                    # projected_reprs[:, 0, :] and projected_reprs[:, 1, :]
                    pred_sim = F.cosine_similarity(projected_reprs[:, 0, :], projected_reprs[:, 1, :], dim=1) # (batch)
                    
                    aux_loss = aux_criterion(pred_sim, target_sim.detach()) # Detach target to not backprop through it for this path
                    loss = loss + aux_relational_loss_weight * aux_loss

            current_batch_loss = loss
            current_batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            batch_loss_item = current_batch_loss.item()
            total_epoch_loss += batch_loss_item
            running_loss_100_steps += batch_loss_item
            num_batches +=1
            
            current_epoch_avg_loss = total_epoch_loss / num_batches
            progress_bar.set_postfix(batch_loss=f"{batch_loss_item:.4f}", epoch_avg_loss=f"{current_epoch_avg_loss:.4f}")

            # Use batch_idx to determine print frequency
            # (batch_idx + 1) gives the 1-indexed current step number
            current_step_in_epoch = batch_idx + 1
            if print_every_n_steps > 0 and current_step_in_epoch % print_every_n_steps == 0:
                avg_loss_last_n = running_loss_100_steps / print_every_n_steps
                tqdm.write(f"  Epoch {epoch + 1}, Step {current_step_in_epoch}/{len(train_loader)}, Avg Loss (last {print_every_n_steps} steps): {avg_loss_last_n:.4f}")
                running_loss_100_steps = 0.0 # Reset for the next block of steps
            
        avg_epoch_loss = total_epoch_loss / num_batches if num_batches > 0 else 0
        progress_bar.close()

def generate_text(model, tokenizer, prompt, max_length=100, temperature=1.0):
    """Generate text using the orthogonal model"""
    model.eval()
    
    input_ids = torch.tensor([tokenizer.encode(prompt)])
    
    is_projected_space = hasattr(model, 'token_embedder') and hasattr(model.token_embedder, 'projection_layer')

    with torch.no_grad():
        for _ in range(max_length):
            if is_projected_space:
                logits, _ = model(input_ids)
            else:
                print("Model structure not recognized for generate_text")
                return
            
            probs = F.softmax(logits[:, -1, :] / temperature, dim=-1)
            next_token = torch.multinomial(probs, 1)
            
            input_ids = torch.cat([input_ids, next_token], dim=1)
            
            if next_token.item() == 0:
                break
    
    return tokenizer.decode(input_ids[0].tolist())

def analyze_orthogonality(model, tokenizer, sample_text="The quick brown fox"):
    """Analyze the orthogonality of embeddings"""
    model.eval()
    
    input_ids = torch.tensor([tokenizer.encode(sample_text)]).to(next(model.parameters()).device)
    
    with torch.no_grad():
        if hasattr(model, 'token_embedder') and hasattr(model.token_embedder, 'projection_layer'):
            standard_embed = model.token_embedder(input_ids, to_projected_space=False)
            projected_embed = model.token_embedder(input_ids, to_projected_space=True)
        elif hasattr(model, 'orthogonal_embedding'):
            standard_embed = model.orthogonal_embedding(input_ids, inverse=False)
            projected_embed = model.orthogonal_embedding(input_ids, inverse=True)
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
        ax2.set_title('Projected Space Dimension Correlations')
        
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
    d_projection_val = 64
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
    train_orthogonal_model(projected_space_model, train_loader, epochs=10, tokenizer=tokenizer)
    
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
