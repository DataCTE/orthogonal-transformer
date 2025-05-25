import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from ml import (OrthogonalNextTokenPredictor, StandardTransformer, 
                CharTokenizer, TextDataset, train_orthogonal_model)
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple
import json
from datasets import load_dataset
import torch.optim.lr_scheduler
from sklearn.decomposition import PCA # Added import

class OrthogonalConceptTester:
    """Test suite for orthogonal token prediction concept"""
    
    def __init__(self, orthogonal_model, standard_model, tokenizer):
        self.orthogonal_model = orthogonal_model
        self.standard_model = standard_model
        self.tokenizer = tokenizer
        # Assuming both models are on the same device, store it from one of them
        self.device = next(orthogonal_model.parameters()).device 
        
    def test_embedding_independence(self, test_sequences: List[str]) -> Dict:
        """Test if orthogonal embeddings are truly independent"""
        results = {
            'sequences': test_sequences,
            'orthogonal_independence': [],
            'standard_independence': []
        }
        
        self.orthogonal_model.eval()
        # Standard model is not directly used here for its own embeddings, but good practice to eval if it were
        self.standard_model.eval() 
        
        with torch.no_grad():
            for seq in test_sequences:
                input_ids = torch.tensor([self.tokenizer.encode(seq)], device=self.device) # Move to device
                
                # Get embeddings
                orth_embed = self.orthogonal_model.orthogonal_embedding(input_ids, inverse=True)
                # For standard_independence, we need embeddings from the standard path of the orthogonal model
                # or embeddings from the actual standard_model. Let's use the orthogonal_model's standard path.
                std_embed_orth_path = self.orthogonal_model.orthogonal_embedding(input_ids, inverse=False)
                
                # Calculate independence metrics (mutual information approximation)
                orth_flat = orth_embed.reshape(-1, self.orthogonal_model.d_model)
                std_flat_orth_path = std_embed_orth_path.reshape(-1, self.orthogonal_model.d_model)
                
                # Compute covariance
                orth_cov = torch.cov(orth_flat.T)
                std_cov_orth_path = torch.cov(std_flat_orth_path.T)
                
                # Independence score (lower is better)
                orth_score = torch.abs(orth_cov - torch.diag(torch.diag(orth_cov))).mean().item()
                std_score = torch.abs(std_cov_orth_path - torch.diag(torch.diag(std_cov_orth_path))).mean().item()
                
                results['orthogonal_independence'].append(orth_score)
                results['standard_independence'].append(std_score) # This was for the orthogonal model's standard path
        
        return results
    
    def test_prediction_diversity(self, prompts: List[str], num_samples: int = 10) -> Dict:
        """Test diversity of predictions"""
        results = {
            'prompts': prompts,
            'orthogonal_diversity': [],
            'standard_diversity': []
        }
        
        self.orthogonal_model.eval()
        self.standard_model.eval()

        for prompt in prompts:
            orth_predictions = []
            std_predictions = []
            
            for _ in range(num_samples):
                # Generate with orthogonal model
                orth_gen = self._generate_tokens(self.orthogonal_model, prompt, 20, temperature=1.0)
                orth_predictions.append(orth_gen)
                
                # Generate with standard model
                std_gen = self._generate_tokens(self.standard_model, prompt, 20, temperature=1.0, is_standard=True)
                std_predictions.append(std_gen)
            
            # Calculate diversity (unique n-grams)
            orth_diversity = self._calculate_diversity(orth_predictions)
            std_diversity = self._calculate_diversity(std_predictions)
            
            results['orthogonal_diversity'].append(orth_diversity)
            results['standard_diversity'].append(std_diversity)
        
        return results
    
    def test_robustness_to_perturbation(self, test_text: str, noise_levels: List[float]) -> Dict:
        """Test model robustness to input perturbations"""
        results = {
            'noise_levels': noise_levels,
            'orthogonal_robustness': [],
            'standard_robustness': []
        }
        
        self.orthogonal_model.eval()
        self.standard_model.eval()

        # Encode original text
        original_ids = torch.tensor([self.tokenizer.encode(test_text)], device=self.device) # Move to device
        
        with torch.no_grad():
            # Get original predictions
            orth_orig_logits, _ = self.orthogonal_model(original_ids)
            std_orig_logits = self.standard_model(original_ids)
            
            for noise_level in noise_levels:
                # Add noise to embeddings
                # For orthogonal model, get its "inverse=True" embeddings
                orth_embed_inv = self.orthogonal_model.orthogonal_embedding(original_ids, inverse=True)
                # For standard model, get its standard embeddings
                std_embed = self.standard_model.embedding(original_ids) # Assuming standard_model has .embedding attribute
                
                # Add Gaussian noise
                orth_noisy_inv = orth_embed_inv + torch.randn_like(orth_embed_inv) * noise_level
                std_noisy = std_embed + torch.randn_like(std_embed) * noise_level
                
                # Forward pass with noisy embeddings
                # _forward_from_embeddings needs to handle the specific embedding types for each model
                # For orthogonal model, we are perturbing its 'inverse' embeddings
                orth_noisy_logits = self._forward_from_embeddings(self.orthogonal_model, orth_noisy_inv, is_standard=False, from_inverse_embedding_for_orth=True)
                std_noisy_logits = self._forward_from_embeddings(self.standard_model, std_noisy, is_standard=True)
                
                # Calculate KL divergence
                orth_kl = F.kl_div(
                    F.log_softmax(orth_noisy_logits, dim=-1),
                    F.softmax(orth_orig_logits, dim=-1),
                    reduction='mean'
                ).item()
                
                std_kl = F.kl_div(
                    F.log_softmax(std_noisy_logits, dim=-1),
                    F.softmax(std_orig_logits, dim=-1),
                    reduction='mean'
                ).item()
                
                results['orthogonal_robustness'].append(orth_kl)
                results['standard_robustness'].append(std_kl)
        
        return results
    
    def test_information_bottleneck(self, test_sequences: List[str]) -> Dict:
        """Test if orthogonal representation acts as information bottleneck"""
        results = {
            'sequences': test_sequences,
            'orthogonal_entropy': [],
            'standard_entropy': [],
            'orthogonal_mutual_info': [],
            'standard_mutual_info': []
        }
        
        self.orthogonal_model.eval()
        self.standard_model.eval()

        with torch.no_grad():
            for seq in test_sequences:
                input_ids = torch.tensor([self.tokenizer.encode(seq)], device=self.device) # Move to device
                
                # Get predictions and embeddings
                orth_logits, inverse_preds = self.orthogonal_model(input_ids) # inverse_preds are from orthogonal_model's inverse path
                std_logits = self.standard_model(input_ids)
                
                # Calculate entropy of predictions
                orth_probs = F.softmax(orth_logits, dim=-1)
                std_probs = F.softmax(std_logits, dim=-1)
                
                orth_entropy = -(orth_probs * torch.log(orth_probs + 1e-8)).sum(dim=-1).mean().item()
                std_entropy = -(std_probs * torch.log(std_probs + 1e-8)).sum(dim=-1).mean().item()
                
                results['orthogonal_entropy'].append(orth_entropy)
                results['standard_entropy'].append(std_entropy)
                
                # Estimate mutual information between input and hidden representations
                orth_mi = self._estimate_mutual_information(input_ids, inverse_preds)
                std_embed = self.standard_model.embedding(input_ids)
                std_mi = self._estimate_mutual_information(input_ids, std_embed)
                
                results['orthogonal_mutual_info'].append(orth_mi)
                results['standard_mutual_info'].append(std_mi)
        
        return results
    
    def visualize_results(self, all_results: Dict):
        """Create comprehensive visualizations"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # 1. Independence scores
        ax = axes[0, 0]
        independence_data = all_results['independence']
        x = range(len(independence_data['sequences']))
        ax.bar(x, independence_data['orthogonal_independence'], alpha=0.7, label='Orthogonal')
        ax.bar(x, independence_data['standard_independence'], alpha=0.7, label='Standard')
        ax.set_xlabel('Sequences')
        ax.set_ylabel('Independence Score (lower is better)')
        ax.set_title('Embedding Independence')
        ax.legend()
        
        # 2. Diversity scores
        ax = axes[0, 1]
        diversity_data = all_results['diversity']
        x = range(len(diversity_data['prompts']))
        width = 0.35
        ax.bar([i - width/2 for i in x], diversity_data['orthogonal_diversity'], 
               width, label='Orthogonal', alpha=0.7)
        ax.bar([i + width/2 for i in x], diversity_data['standard_diversity'], 
               width, label='Standard', alpha=0.7)
        ax.set_xlabel('Prompts')
        ax.set_ylabel('Diversity Score')
        ax.set_title('Prediction Diversity')
        ax.legend()
        
        # 3. Robustness
        ax = axes[0, 2]
        robustness_data = all_results['robustness']
        ax.plot(robustness_data['noise_levels'], robustness_data['orthogonal_robustness'], 
                'o-', label='Orthogonal', linewidth=2)
        ax.plot(robustness_data['noise_levels'], robustness_data['standard_robustness'], 
                's-', label='Standard', linewidth=2)
        ax.set_xlabel('Noise Level')
        ax.set_ylabel('KL Divergence')
        ax.set_title('Robustness to Perturbation')
        ax.legend()
        
        # 4. Entropy
        ax = axes[1, 0]
        bottleneck_data = all_results['bottleneck']
        x = range(len(bottleneck_data['sequences']))
        ax.scatter(x, bottleneck_data['orthogonal_entropy'], label='Orthogonal', alpha=0.7, s=100)
        ax.scatter(x, bottleneck_data['standard_entropy'], label='Standard', alpha=0.7, s=100)
        ax.set_xlabel('Sequences')
        ax.set_ylabel('Entropy')
        ax.set_title('Prediction Entropy')
        ax.legend()
        
        # 5. Mutual Information
        ax = axes[1, 1]
        ax.bar([i - width/2 for i in x], bottleneck_data['orthogonal_mutual_info'], 
               width, label='Orthogonal', alpha=0.7)
        ax.bar([i + width/2 for i in x], bottleneck_data['standard_mutual_info'], 
               width, label='Standard', alpha=0.7)
        ax.set_xlabel('Sequences')
        ax.set_ylabel('Mutual Information')
        ax.set_title('Input-Hidden Mutual Information')
        ax.legend()
        
        # 6. Summary metrics
        ax = axes[1, 2]
        metrics = {
            'Independence': (
                np.mean(independence_data['orthogonal_independence']),
                np.mean(independence_data['standard_independence'])
            ),
            'Diversity': (
                np.mean(diversity_data['orthogonal_diversity']),
                np.mean(diversity_data['standard_diversity'])
            ),
            'Robustness': (
                np.mean(robustness_data['orthogonal_robustness']),
                np.mean(robustness_data['standard_robustness'])
            ),
            'Entropy': (
                np.mean(bottleneck_data['orthogonal_entropy']),
                np.mean(bottleneck_data['standard_entropy'])
            )
        }
        
        metric_names = list(metrics.keys())
        orth_values = [metrics[m][0] for m in metric_names]
        std_values = [metrics[m][1] for m in metric_names]
        
        x = range(len(metric_names))
        ax.bar([i - width/2 for i in x], orth_values, width, label='Orthogonal', alpha=0.7)
        ax.bar([i + width/2 for i in x], std_values, width, label='Standard', alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(metric_names, rotation=45)
        ax.set_ylabel('Average Score')
        ax.set_title('Summary Comparison')
        ax.legend()
        
        plt.tight_layout()
        plt.savefig('orthogonal_concept_analysis.png', dpi=150)
        plt.close()
        print("Visualizations saved to 'orthogonal_concept_analysis.png'") # Added print statement
    
    def visualize_embedding_point_cloud(self, words_for_cloud: List[str], filename: str = "orthogonal_embedding_point_cloud.png"):
        """
        Visualizes a 2D PCA projection of character embeddings from specified words
        for both standard (pre-orthogonal transformation) and inverse/orthogonal space.
        """
        self.orthogonal_model.eval()
        
        all_standard_embeds_list = []
        all_inverse_embeds_list = []
        all_labels = []
        
        print(f"\nGenerating point cloud for words (orthogonal model): {words_for_cloud}")

        with torch.no_grad():
            for word in words_for_cloud:
                if not word: continue
                input_ids = torch.tensor([self.tokenizer.encode(word)], device=self.device)
                if input_ids.nelement() == 0: continue

                # Get standard embeddings (inverse=False)
                standard_embeds_word = self.orthogonal_model.orthogonal_embedding(input_ids, inverse=False)
                # Get inverse/orthogonal embeddings (inverse=True)
                inverse_embeds_word = self.orthogonal_model.orthogonal_embedding(input_ids, inverse=True)

                for i, token_id in enumerate(input_ids[0]):
                    char_label = self.tokenizer.decode([token_id.item()])
                    all_labels.append(f"{char_label}({word[:2]})") 

                    all_standard_embeds_list.append(standard_embeds_word[0, i, :].unsqueeze(0))
                    all_inverse_embeds_list.append(inverse_embeds_word[0, i, :].unsqueeze(0))
        
        if not all_standard_embeds_list or not all_inverse_embeds_list:
            print("No embeddings generated for point cloud. Skipping visualization.")
            return

        all_standard_embeds = torch.cat(all_standard_embeds_list, dim=0).cpu().numpy()
        all_inverse_embeds = torch.cat(all_inverse_embeds_list, dim=0).cpu().numpy()

        if all_standard_embeds.shape[0] < 2 or all_inverse_embeds.shape[0] < 2:
            print("Not enough data points for PCA. Skipping point cloud visualization.")
            return

        pca_standard = PCA(n_components=2, random_state=42)
        standard_2d = pca_standard.fit_transform(all_standard_embeds)
        
        pca_inverse = PCA(n_components=2, random_state=42)
        inverse_2d = pca_inverse.fit_transform(all_inverse_embeds)
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        fig.suptitle("Character Embedding Point Cloud (Orthogonal Model - PCA Reduced)", fontsize=16)

        axes[0].scatter(standard_2d[:, 0], standard_2d[:, 1], alpha=0.7)
        for i, label in enumerate(all_labels):
            axes[0].annotate(label, (standard_2d[i, 0], standard_2d[i, 1]), fontsize=8)
        axes[0].set_title(f'Standard Embeddings (d_model={self.orthogonal_model.d_model})')
        axes[0].set_xlabel('PCA Component 1')
        axes[0].set_ylabel('PCA Component 2')
        axes[0].grid(True, linestyle='--', alpha=0.5)

        axes[1].scatter(inverse_2d[:, 0], inverse_2d[:, 1], alpha=0.7, color='green')
        for i, label in enumerate(all_labels):
            axes[1].annotate(label, (inverse_2d[i, 0], inverse_2d[i, 1]), fontsize=8)
        axes[1].set_title(f'Inverse/Orthogonal Embeddings (d_model={self.orthogonal_model.d_model})')
        axes[1].set_xlabel('PCA Component 1')
        axes[1].set_ylabel('PCA Component 2')
        axes[1].grid(True, linestyle='--', alpha=0.5)
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(filename, dpi=150)
        plt.close()
        print(f"Point cloud visualization saved to '{filename}'")
    
    def _generate_tokens(self, model, prompt, max_length, temperature, is_standard=False):
        """Helper to generate tokens"""
        input_ids = torch.tensor([self.tokenizer.encode(prompt)], device=self.device) # Move to device
        
        model.eval() # Ensure model is in eval mode for generation
        with torch.no_grad():
            for _ in range(max_length):
                if is_standard:
                    logits = model(input_ids)
                else:
                    logits, _ = model(input_ids)
                
                probs = F.softmax(logits[:, -1, :] / temperature, dim=-1)
                next_token = torch.multinomial(probs, 1)
                input_ids = torch.cat([input_ids, next_token], dim=1)
        
        return self.tokenizer.decode(input_ids[0].tolist())
    
    def _calculate_diversity(self, texts, n=3):
        """Calculate n-gram diversity"""
        all_ngrams = set()
        for text in texts:
            for i in range(len(text) - n + 1):
                all_ngrams.add(text[i:i+n])
        return len(all_ngrams) / max(1, sum(len(text) - n + 1 for text in texts))
    
    def _forward_from_embeddings(self, model, embeddings, is_standard=False, from_inverse_embedding_for_orth=False): # Added from_inverse_embedding_for_orth
        """Forward pass from embeddings"""
        seq_len = embeddings.size(1)
        # Determine the current device from the input embeddings, as that's what we need to align with.
        current_device = embeddings.device
        
        # Add positional encoding
        # Ensure the positional encoding slice is on the same device as embeddings.
        pe_slice = model.positional_encoding[:, :seq_len, :].to(current_device)
        x = embeddings + pe_slice
        x = model.dropout(x) # Dropout is part of the model
        
        # Create mask on the same device as the data it will mask.
        mask = model.create_causal_mask(seq_len, current_device) 
        
        # Pass through transformer
        for block in model.transformer_blocks:
            x = block(x, mask)
        
        x = model.norm(x)
        
        if is_standard: # This is for the StandardTransformer model
            return model.output_projection(x)
        else: # This is for the OrthogonalNextTokenPredictor model
            # Project to its output space (which is d_model for inverse_predictions)
            output_repr = model.output_projection(x)

            if from_inverse_embedding_for_orth:
                # If the input `embeddings` were already inverse_embeddings for the orthogonal model,
                # then output_repr is already in the inverse prediction space.
                # We then convert these inverse_predictions back to token logits.
                all_inverse_embeddings = model.orthogonal_embedding.embedding.weight @ \
                                        (model.orthogonal_embedding.transform_weight @ 
                                         model.orthogonal_embedding.orthogonal_matrix).T
                return torch.matmul(output_repr, all_inverse_embeddings.T)
            else:
                # This case would be if we fed standard embeddings to the orthogonal model's
                # _forward_from_embeddings, which is not what test_robustness_to_perturbation is doing.
                # For robustness, we perturb the *inverse* embeddings.
                # If `embeddings` were standard embeddings, this path would need different handling.
                # However, for the current usage, from_inverse_embedding_for_orth should be True.
                # For safety, let's assume this path also converts from the model's internal "inverse prediction space" to logits
                all_inverse_embeddings = model.orthogonal_embedding.embedding.weight @ \
                                        (model.orthogonal_embedding.transform_weight @ 
                                         model.orthogonal_embedding.orthogonal_matrix).T
                return torch.matmul(output_repr, all_inverse_embeddings.T)
    
    def _estimate_mutual_information(self, input_ids, hidden_repr):
        """Estimate mutual information using binning"""
        # Simple estimation using histogram binning
        input_flat = input_ids.reshape(-1).cpu().numpy()
        hidden_flat = hidden_repr.mean(dim=-1).reshape(-1).cpu().detach().numpy()
        
        # Discretize continuous values
        hidden_bins = np.histogram_bin_edges(hidden_flat, bins=10)
        hidden_discrete = np.digitize(hidden_flat, hidden_bins)
        
        # Calculate joint and marginal entropies
        joint_hist, _, _ = np.histogram2d(input_flat, hidden_discrete, bins=10)
        joint_hist = joint_hist / joint_hist.sum()
        
        marginal_input = joint_hist.sum(axis=1)
        marginal_hidden = joint_hist.sum(axis=0)
        
        # MI = H(X) + H(Y) - H(X,Y)
        h_input = -np.sum(marginal_input * np.log(marginal_input + 1e-8))
        h_hidden = -np.sum(marginal_hidden * np.log(marginal_hidden + 1e-8))
        h_joint = -np.sum(joint_hist * np.log(joint_hist + 1e-8))
        
        return h_input + h_hidden - h_joint

def calculate_perplexity(model, data_loader, criterion, device, is_standard_model=False, tokenizer=None):
    """Calculates perplexity for a given model and data_loader."""
    model.eval()
    total_loss = 0
    total_tokens = 0
    
    pad_token_id = 0 # Assuming pad token ID is 0, as used in CrossEntropyLoss ignore_index
    if tokenizer and hasattr(tokenizer, 'pad_id') and tokenizer.pad_id is not None:
        pad_token_id = tokenizer.pad_id

    with torch.no_grad():
        for batch in data_loader:
            # Assuming data_loader yields (input_ids, target_ids)
            # Handle cases where data_loader might yield a dictionary or other structures if TextDataset changes
            if isinstance(batch, (list, tuple)) and len(batch) == 2:
                input_ids, target_ids = batch
            else: # Fallback assuming batch is input_ids and targets are derived or handled by model
                input_ids = batch
                target_ids = input_ids[:, 1:].contiguous() # Common setup, but ensure TextDataset provides this
                input_ids = input_ids[:, :-1].contiguous()


            input_ids, target_ids = input_ids.to(device), target_ids.to(device)
            
            if is_standard_model:
                logits = model(input_ids)
            else:
                logits, _ = model(input_ids) # Orthogonal model returns (main_logits, inverse_logits)
            
            # Calculate loss only on non-padding tokens
            # Reshape logits to (batch_size * seq_len, vocab_size)
            # Reshape target_ids to (batch_size * seq_len)
            loss = criterion(logits.view(-1, logits.size(-1)), target_ids.view(-1))

            # To correctly calculate perplexity, we need sum of losses and total token count.
            # The criterion with reduction='mean' already averages over non-ignored tokens in the batch.
            # So, total_loss should accumulate this mean loss multiplied by the number of active tokens.
            
            active_tokens_in_batch = (target_ids.view(-1) != pad_token_id).sum().item()
            if active_tokens_in_batch > 0:
                # CrossEntropyLoss with ignore_index and reduction='mean' calculates mean loss over non-padded tokens.
                # To get sum of losses for perplexity, we calculate loss with reduction='sum'.
                current_loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)), 
                    target_ids.view(-1), 
                    ignore_index=pad_token_id, 
                    reduction='sum'
                )
                total_loss += current_loss.item()
                total_tokens += active_tokens_in_batch
                
    if total_tokens == 0:
        print("Warning: No tokens found for perplexity calculation. Returning inf.")
        return float('inf')
        
    avg_loss = total_loss / total_tokens
    perplexity = np.exp(avg_loss)
    model.train() # Set model back to training mode
    return perplexity

def run_comprehensive_tests():
    """Run all tests"""
    print("Loading models and data...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Hugging Face Dataset Configuration ---
    dataset_name = "wikitext"
    dataset_config = "wikitext-2-raw-v1" # Use 'wikitext-103-raw-v1' for a much larger dataset
    max_train_samples = 5000  # Limit number of samples for quicker run, adjust as needed
    max_val_samples = 500    # For perplexity calculation
    train_max_length = 128      # Max sequence length for training
    eval_max_length = 128       # Max sequence length for perplexity evaluation

    print(f"Loading dataset: {dataset_name} ({dataset_config})")
    raw_datasets = load_dataset(dataset_name, dataset_config)

    # Process text data
    train_texts = [text for text in raw_datasets['train']['text'] if text.strip()][:max_train_samples]
    val_texts = [text for text in raw_datasets['validation']['text'] if text.strip()][:max_val_samples]

    if not train_texts:
        raise ValueError("No training texts loaded. Check dataset and filtering.")
    if not val_texts:
        print("Warning: No validation texts loaded. Perplexity will not be calculated.")
        # Fallback to using a small portion of train_texts if val_texts is empty and needed
        if not val_texts and max_val_samples > 0:
            val_texts = train_texts[:min(len(train_texts), max_val_samples)]


    # Initialize tokenizer
    tokenizer = CharTokenizer()
    print("Building tokenizer vocab on training data...")
    tokenizer.build_vocab(train_texts + val_texts) # Build vocab on combined texts to ensure coverage
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    
    # Create dataset and dataloader
    print("Creating datasets and dataloaders...")
    train_dataset = TextDataset(train_texts, tokenizer, max_length=train_max_length)
    val_dataset = TextDataset(val_texts, tokenizer, max_length=eval_max_length)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2, pin_memory=True if device.type == 'cuda' else False)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2, pin_memory=True if device.type == 'cuda' else False)
    
    # Initialize models
    model_params = dict(
        vocab_size=tokenizer.vocab_size,
        d_model=256,        # Increased model size
        n_heads=8,          # Increased model size
        n_layers=4,         # Increased model size
        d_ff=1024,         # Increased model size
        max_seq_len=train_max_length, # Adjusted to training max length
        dropout=0.1         # Added dropout
    )
    
    orthogonal_model = OrthogonalNextTokenPredictor(**model_params).to(device)
    standard_model = StandardTransformer(**model_params).to(device)
    
    # Train models
    epochs = 10 # Adjust epochs based on dataset size and desired training time
    lr = 1e-4   # Adjusted learning rate

    print(f"Training Orthogonal Model for {epochs} epochs...")
    train_orthogonal_model(orthogonal_model, train_loader, epochs=epochs, lr=lr, device=device, tokenizer=tokenizer)
    
    print(f"\nTraining Standard Model for {epochs} epochs...")
    optimizer = torch.optim.Adam(standard_model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=epochs//3, gamma=0.1) # Added scheduler
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_id if hasattr(tokenizer, 'pad_id') else 0) # Use tokenizer.pad_id
    
    standard_model.train()
    for epoch in range(epochs):
        epoch_loss = 0
        num_batches = 0
        for input_ids, target_ids in train_loader:
            input_ids, target_ids = input_ids.to(device), target_ids.to(device)
            optimizer.zero_grad()
            logits = standard_model(input_ids)
            loss = criterion(logits.reshape(-1, logits.size(-1)), target_ids.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(standard_model.parameters(), 1.0) # Grad clipping
            optimizer.step()
            epoch_loss += loss.item()
            num_batches += 1
        scheduler.step()
        avg_epoch_loss = epoch_loss / num_batches
        print(f"Standard Model - Epoch {epoch+1}/{epochs}, Avg Loss: {avg_epoch_loss:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")
    
    # --- Perplexity Calculation ---
    print("\nCalculating perplexity on validation set...")
    # Ensure criterion for perplexity calculation ignores padding
    perplexity_criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_id if hasattr(tokenizer, 'pad_id') else 0, reduction='mean')

    orth_perplexity = float('inf')
    std_perplexity = float('inf')

    if val_texts: # Only calculate if validation data is available
        orth_perplexity = calculate_perplexity(orthogonal_model, val_loader, perplexity_criterion, device, is_standard_model=False, tokenizer=tokenizer)
        std_perplexity = calculate_perplexity(standard_model, val_loader, perplexity_criterion, device, is_standard_model=True, tokenizer=tokenizer)
        print(f"Orthogonal Model Perplexity: {orth_perplexity:.2f}")
        print(f"Standard Model Perplexity: {std_perplexity:.2f}")
    else:
        print("Skipping perplexity calculation as no validation data was loaded.")

    # Initialize tester
    tester = OrthogonalConceptTester(orthogonal_model, standard_model, tokenizer)
    
    print("\nRunning comprehensive tests...")
    
    # Test 1: Independence
    print("Testing embedding independence...")
    test_sequences = ["The quick brown fox", "Machine learning models", "Neural networks are", "Deep learning has"] # Slightly longer
    # Ensure test_sequences are tokenizable by the new tokenizer
    independence_results = tester.test_embedding_independence(test_sequences)
    
    # Test 2: Diversity
    print("Testing prediction diversity...")
    prompts = ["The ", "Machine learning ", "Neural network is ", "Deep learning can "]
    diversity_results = tester.test_prediction_diversity(prompts, num_samples=10) # Reduced samples for speed
    
    # Test 3: Robustness
    print("Testing robustness to perturbation...")
    noise_levels = [0.0, 0.05, 0.1, 0.2, 0.5] # Adjusted noise levels
    robustness_results = tester.test_robustness_to_perturbation(
        "The quick brown fox jumps over the lazy dog", noise_levels
    )
    
    # Test 4: Information bottleneck
    print("Testing information bottleneck properties...")
    bottleneck_results = tester.test_information_bottleneck(test_sequences) # Use same sequences
    
    # Compile all results
    all_results = {
        'meta_info': {
            'dataset_name': dataset_name,
            'dataset_config': dataset_config,
            'max_train_samples': max_train_samples,
            'max_val_samples': max_val_samples,
            'train_max_length': train_max_length,
            'epochs': epochs,
            'learning_rate': lr,
            'model_params': model_params,
            'orthogonal_perplexity': orth_perplexity if val_texts else 'N/A',
            'standard_perplexity': std_perplexity if val_texts else 'N/A',
        },
        'independence': independence_results,
        'diversity': diversity_results,
        'robustness': robustness_results,
        'bottleneck': bottleneck_results
    }
    
    # Save results
    results_filename = 'orthogonal_test_results_large_run.json'
    print(f"\nSaving detailed results to '{results_filename}'")
    with open(results_filename, 'w') as f:
        # Custom JSON encoder for handling potential numpy types or float('inf')
        class NpEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                if isinstance(obj, np.floating):
                    return float(obj)
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                if obj == float('inf'):
                    return "Infinity"
                return super(NpEncoder, self).default(obj)
        json.dump(all_results, f, indent=2, cls=NpEncoder)
    
    # Visualize
    print("\nGenerating visualizations...")
    tester.visualize_results(all_results) # This might need adjustment if keys change
    
    # Define words for point cloud visualization
    point_cloud_words = [
        "king", "queen", "man", "woman", 
        "apple", "orange", "fruit", 
        "happy", "sad", "joy", "fear",
        "run", "walk", "fast", "slow"
    ]
    tester.visualize_embedding_point_cloud(point_cloud_words) # Call the new method
    
    # Print summary
    print("\n" + "="*50)
    print("COMPREHENSIVE TEST SUMMARY (Large Run)")
    print("="*50)

    if val_texts:
        print(f"\nPERPLEXITY (on {dataset_name} validation - {max_val_samples} samples):")
        print(f"   Orthogonal Model: {orth_perplexity:.2f}")
        print(f"   Standard Model:   {std_perplexity:.2f}")
    
    print("\n1. EMBEDDING INDEPENDENCE:")
    avg_orth_ind = np.mean(independence_results['orthogonal_independence'])
    avg_std_ind = np.mean(independence_results['standard_independence'])
    print(f"   Orthogonal: {avg_orth_ind:.4f}")
    print(f"   Standard:   {avg_std_ind:.4f}")
    print(f"   Improvement: {(avg_std_ind - avg_orth_ind) / avg_std_ind * 100:.1f}%")
    
    print("\n2. PREDICTION DIVERSITY:")
    avg_orth_div = np.mean(diversity_results['orthogonal_diversity'])
    avg_std_div = np.mean(diversity_results['standard_diversity'])
    print(f"   Orthogonal: {avg_orth_div:.4f}")
    print(f"   Standard:   {avg_std_div:.4f}")
    print(f"   Improvement: {(avg_orth_div - avg_std_div) / avg_std_div * 100:.1f}%")
    
    print("\n3. ROBUSTNESS (avg KL divergence):")
    avg_orth_rob = np.mean(robustness_results['orthogonal_robustness'])
    avg_std_rob = np.mean(robustness_results['standard_robustness'])
    print(f"   Orthogonal: {avg_orth_rob:.4f}")
    print(f"   Standard:   {avg_std_rob:.4f}")
    print(f"   Improvement: {(avg_std_rob - avg_orth_rob) / avg_std_rob * 100:.1f}%")
    
    print("\n4. INFORMATION PROPERTIES:")
    avg_orth_ent = np.mean(bottleneck_results['orthogonal_entropy'])
    avg_std_ent = np.mean(bottleneck_results['standard_entropy'])
    avg_orth_mi = np.mean(bottleneck_results['orthogonal_mutual_info']) # MI can be NaN if histograms are empty
    avg_std_mi = np.mean(bottleneck_results['standard_mutual_info'])
    
    # Handle potential NaNs from MI calculation if dataset is too small or uniform
    avg_orth_mi = np.nan_to_num(avg_orth_mi, nan=0.0)
    avg_std_mi = np.nan_to_num(avg_std_mi, nan=0.0)

    print(f"   Entropy - Orthogonal: {avg_orth_ent:.4f}, Standard: {avg_std_ent:.4f}")
    print(f"   Mutual Info - Orthogonal: {avg_orth_mi:.4f}, Standard: {avg_std_mi:.4f}")
    
    print(f"\nAnalysis complete! Check 'orthogonal_concept_analysis.png' for visualizations.")
    print(f"Detailed results saved to '{results_filename}'")

if __name__ == "__main__":
    run_comprehensive_tests() 