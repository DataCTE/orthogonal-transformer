import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from projected_space_transformer_model import (ProjectedSpaceTransformer, StandardTransformer, 
                             CharTokenizer, TextDataset, train_projected_space_model)
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple
import json
from datasets import load_dataset
import torch.optim.lr_scheduler
from sklearn.decomposition import PCA

class ProjectedSpaceConceptTester:
    """Test suite for projected space token prediction concept"""
    
    def __init__(self, projected_space_model, standard_model, tokenizer):
        self.projected_space_model = projected_space_model
        self.standard_model = standard_model
        self.tokenizer = tokenizer
        self.device = next(projected_space_model.parameters()).device
        
    def test_embedding_independence(self, test_sequences: List[str]) -> Dict:
        """Test if projected space embeddings show different independence characteristics"""
        results = {
            'sequences': test_sequences,
            'projected_space_independence': [],
            'd_model_pre_projection_independence': []
        }
        
        self.projected_space_model.eval()
        self.standard_model.eval() 
        
        with torch.no_grad():
            for seq in test_sequences:
                input_ids = torch.tensor([self.tokenizer.encode(seq)], device=self.device)
                
                # Get projected embeddings (d_projection)
                projected_embed = self.projected_space_model.token_embedder(input_ids, to_projected_space=True)
                # Get d_model embeddings (before projection)
                d_model_embed = self.projected_space_model.token_embedder(input_ids, to_projected_space=False)
                
                projected_flat = projected_embed.reshape(-1, self.projected_space_model.d_projection)
                
                if projected_flat.shape[0] < 2 or projected_flat.shape[1] < 2:
                    projected_score = float('nan')
                else:
                    projected_cov = torch.cov(projected_flat.T)
                    projected_score = torch.abs(projected_cov - torch.diag(torch.diag(projected_cov))).mean().item()
                
                results['projected_space_independence'].append(projected_score)
        
        return results
    
    def test_prediction_diversity(self, prompts: List[str], num_samples: int = 10) -> Dict:
        results = {
            'prompts': prompts,
            'projected_space_diversity': [],
            'standard_diversity': []
        }
        
        self.projected_space_model.eval()
        self.standard_model.eval()

        for prompt in prompts:
            projected_space_predictions = []
            std_predictions = []
            
            for _ in range(num_samples):
                # Generate with projected_space model
                ms_gen = self._generate_tokens(self.projected_space_model, prompt, 20, temperature=1.0, is_standard=False)
                projected_space_predictions.append(ms_gen)
                
                # Generate with standard model
                std_gen = self._generate_tokens(self.standard_model, prompt, 20, temperature=1.0, is_standard=True)
                std_predictions.append(std_gen)
            
            ms_diversity = self._calculate_diversity(projected_space_predictions)
            std_diversity = self._calculate_diversity(std_predictions)
            
            results['projected_space_diversity'].append(ms_diversity)
            results['standard_diversity'].append(std_diversity)
        
        return results
    
    def test_robustness_to_perturbation(self, test_text: str, noise_levels: List[float]) -> Dict:
        results = {
            'noise_levels': noise_levels,
            'projected_space_robustness': [],
            'standard_robustness': []
        }
        
        self.projected_space_model.eval()
        self.standard_model.eval()

        original_ids = torch.tensor([self.tokenizer.encode(test_text)], device=self.device)
        
        with torch.no_grad():
            ms_orig_logits, _ = self.projected_space_model(original_ids)
            std_orig_logits = self.standard_model(original_ids)
            
            for noise_level in noise_levels:
                # For projected_space model, get its projected embeddings (d_projection)
                ms_embed_projection = self.projected_space_model.token_embedder(original_ids, to_projected_space=True)
                std_embed = self.standard_model.embedding(original_ids)
                
                ms_noisy_projection = ms_embed_projection + torch.randn_like(ms_embed_projection) * noise_level
                std_noisy = std_embed + torch.randn_like(std_embed) * noise_level
                
                # Forward pass with noisy embeddings
                # For projected_space model, we are perturbing its d_projection embeddings.
                # The is_standard flag will be False. The from_inverse_embedding_for_orth is not relevant here.
                ms_noisy_logits = self._forward_from_embeddings(self.projected_space_model, ms_noisy_projection, is_standard=False)
                std_noisy_logits = self._forward_from_embeddings(self.standard_model, std_noisy, is_standard=True)
                
                ms_kl = F.kl_div(
                    F.log_softmax(ms_noisy_logits, dim=-1),
                    F.softmax(ms_orig_logits, dim=-1),
                    reduction='batchmean'
                ).item()
                
                std_kl = F.kl_div(
                    F.log_softmax(std_noisy_logits, dim=-1),
                    F.softmax(std_orig_logits, dim=-1),
                    reduction='batchmean'
                ).item()
                
                results['projected_space_robustness'].append(ms_kl)
                results['standard_robustness'].append(std_kl)
        
        return results
    
    def test_information_bottleneck(self, test_sequences: List[str]) -> Dict:
        results = {
            'sequences': test_sequences,
            'projected_space_entropy': [],
            'standard_entropy': [],
            'projected_space_mutual_info': [],
            'standard_mutual_info': []
        }
        
        self.projected_space_model.eval()
        self.standard_model.eval()

        with torch.no_grad():
            for seq in test_sequences:
                input_ids = torch.tensor([self.tokenizer.encode(seq)], device=self.device)
                
                ms_logits, projected_reprs = self.projected_space_model(input_ids)
                std_logits = self.standard_model(input_ids)
                
                ms_probs = F.softmax(ms_logits, dim=-1)
                std_probs = F.softmax(std_logits, dim=-1)
                
                ms_entropy = -(ms_probs * torch.log(ms_probs + 1e-8)).sum(dim=-1).mean().item()
                std_entropy = -(std_probs * torch.log(std_probs + 1e-8)).sum(dim=-1).mean().item()
                
                results['projected_space_entropy'].append(ms_entropy)
                results['standard_entropy'].append(std_entropy)
                
                # Estimate mutual information between input and projected_reprs (d_projection)
                ms_mi = self._estimate_mutual_information(input_ids, projected_reprs)
                
                # For standard model, use its d_model embeddings
                std_embed_hidden = self.standard_model.embedding(input_ids) 
                # We need to pass it through transformer blocks to get a comparable "hidden_repr"
                pe_slice_std = self.standard_model.positional_encoding[:, :input_ids.size(1), :].to(self.device)
                std_hidden_x = std_embed_hidden * np.sqrt(self.standard_model.d_model) + pe_slice_std
                std_hidden_x = self.standard_model.dropout(std_hidden_x)
                mask_std = self.standard_model.create_causal_mask(input_ids.size(1), self.device)
                for block in self.standard_model.transformer_blocks:
                    std_hidden_x = block(std_hidden_x, mask_std)
                std_hidden_repr = self.standard_model.norm(std_hidden_x)

                std_mi = self._estimate_mutual_information(input_ids, std_hidden_repr)
                
                results['projected_space_mutual_info'].append(ms_mi)
                results['standard_mutual_info'].append(std_mi)
        
        return results
    
    def visualize_results(self, all_results: Dict):
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle("ProjectedSpace vs Standard Model Analysis", fontsize=16)
        
        # 1. Independence scores
        ax = axes[0, 0]
        independence_data = all_results['independence']
        x_labels_ind = [f"Seq {i+1}" for i in range(len(independence_data['sequences']))]
        x_pos_ind = np.arange(len(x_labels_ind))
        width = 0.35
        
        ax.bar(x_pos_ind - width/2, independence_data['projected_space_independence'], width, alpha=0.7, label='ProjectedSpace (d_projection)')
        ax.bar(x_pos_ind + width/2, independence_data['d_model_pre_projection_independence'], width, alpha=0.7, label='MS Pre-Projection (d_model)')
        ax.set_xlabel('Test Sequences')
        ax.set_ylabel('Avg. Off-Diagonal Covariance (lower is better)')
        ax.set_title('Embedding Space Independence')
        ax.set_xticks(x_pos_ind)
        ax.set_xticklabels(x_labels_ind, rotation=45, ha="right")
        ax.legend()
        
        # 2. Diversity scores
        ax = axes[0, 1]
        diversity_data = all_results['diversity']
        x_labels_div = [f"Prompt {i+1}" for i in range(len(diversity_data['prompts']))]
        x_pos_div = np.arange(len(x_labels_div))
        
        ax.bar(x_pos_div - width/2, diversity_data['projected_space_diversity'], width, label='ProjectedSpace', alpha=0.7)
        ax.bar(x_pos_div + width/2, diversity_data['standard_diversity'], width, label='Standard', alpha=0.7)
        ax.set_xlabel('Prompts')
        ax.set_ylabel('Unique N-gram Ratio (higher is better)')
        ax.set_title('Prediction Diversity')
        ax.set_xticks(x_pos_div)
        ax.set_xticklabels(x_labels_div, rotation=45, ha="right")
        ax.legend()
        
        # 3. Robustness
        ax = axes[0, 2]
        robustness_data = all_results['robustness']
        ax.plot(robustness_data['noise_levels'], robustness_data['projected_space_robustness'], 
                'o-', label='ProjectedSpace', linewidth=2, markersize=8)
        ax.plot(robustness_data['noise_levels'], robustness_data['standard_robustness'], 
                's-', label='Standard', linewidth=2, markersize=8)
        ax.set_xlabel('Input Embedding Noise Level')
        ax.set_ylabel('KL Divergence (lower is better)')
        ax.set_title('Robustness to Perturbation')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # 4. Entropy
        ax = axes[1, 0]
        bottleneck_data = all_results['bottleneck']
        x_labels_entropy = [f"Seq {i+1}" for i in range(len(bottleneck_data['sequences']))]
        x_pos_entropy = np.arange(len(x_labels_entropy))

        ax.bar(x_pos_entropy - width/2, bottleneck_data['projected_space_entropy'], width, label='ProjectedSpace', alpha=0.7)
        ax.bar(x_pos_entropy + width/2, bottleneck_data['standard_entropy'], width, label='Standard', alpha=0.7)
        ax.set_xlabel('Test Sequences')
        ax.set_ylabel('Prediction Entropy (higher is more uncertain/diverse)')
        ax.set_title('Output Prediction Entropy')
        ax.set_xticks(x_pos_entropy)
        ax.set_xticklabels(x_labels_entropy, rotation=45, ha="right")
        ax.legend()
        
        # 5. Mutual Information
        ax = axes[1, 1]
        x_labels_mi = [f"Seq {i+1}" for i in range(len(bottleneck_data['sequences']))]
        x_pos_mi = np.arange(len(x_labels_mi))
        ax.bar(x_pos_mi - width/2, bottleneck_data['projected_space_mutual_info'], 
               width, label='ProjectedSpace (Input to Projection)', alpha=0.7)
        ax.bar(x_pos_mi + width/2, bottleneck_data['standard_mutual_info'], 
               width, label='Standard (Input to Hidden)', alpha=0.7)
        ax.set_xlabel('Test Sequences')
        ax.set_ylabel('Estimated Mutual Information')
        ax.set_title('Input vs. Hidden Representation MI')
        ax.set_xticks(x_pos_mi)
        ax.set_xticklabels(x_labels_mi, rotation=45, ha="right")
        ax.legend()
        
        # 6. Summary metrics (placeholder, can be more sophisticated)
        ax = axes[1, 2]
        summary_metrics = {
            'MS Indep.': np.nanmean(independence_data['projected_space_independence']),
            'Std Indep. (d_model)': np.nanmean(independence_data['d_model_pre_projection_independence']),
            'MS Div.': np.nanmean(diversity_data['projected_space_diversity']),
            'Std Div.': np.nanmean(diversity_data['standard_diversity']),
            'MS Robust.': np.nanmean(robustness_data['projected_space_robustness']),
            'Std Robust.': np.nanmean(robustness_data['standard_robustness']),
            'MS Entropy': np.nanmean(bottleneck_data['projected_space_entropy']),
            'Std Entropy': np.nanmean(bottleneck_data['standard_entropy']),
        }
        metric_names = list(summary_metrics.keys())
        metric_values = list(summary_metrics.values())
        
        y_pos_summary = np.arange(len(metric_names))
        ax.barh(y_pos_summary, metric_values, color=['skyblue', 'lightcoral'] * (len(metric_names)//2 + 1))
        ax.set_yticks(y_pos_summary)
        ax.set_yticklabels(metric_names)
        ax.set_xlabel('Average Score (context dependent)')
        ax.set_title('Overall Metric Averages')
        ax.invert_yaxis()
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig('projected_space_analysis.png', dpi=150)
        plt.close()
        print("Visualizations saved to 'projected_space_analysis.png'")

    def visualize_embedding_point_cloud(self, words_for_cloud: List[str], filename: str = "projected_space_embedding_point_cloud.png"):
        """
        Visualizes a 2D PCA projection of character embeddings from specified words
        for both standard d_model space and d_projection space.
        """
        self.projected_space_model.eval()
        
        all_standard_embeds_list = []
        all_projection_embeds_list = []
        all_labels = []
        
        print(f"\nGenerating point cloud for words: {words_for_cloud}")

        with torch.no_grad():
            for word in words_for_cloud:
                if not word: continue
                input_ids = torch.tensor([self.tokenizer.encode(word)], device=self.device)
                if input_ids.nelement() == 0: continue

                # Get d_model embeddings (pre-projection to projection)
                standard_embeds_word = self.projected_space_model.token_embedder(input_ids, to_projected_space=False)
                # Get projected embeddings (d_projection)
                projection_embeds_word = self.projected_space_model.token_embedder(input_ids, to_projected_space=True)

                for i, token_id in enumerate(input_ids[0]):
                    char_label = self.tokenizer.decode([token_id.item()])
                    # Add a unique identifier if char_label is repeated across words to distinguish them in plot
                    all_labels.append(f"{char_label}({word[:2]})") # e.g., "a(ki)" for 'a' in 'king'

                    all_standard_embeds_list.append(standard_embeds_word[0, i, :].unsqueeze(0))
                    all_projection_embeds_list.append(projection_embeds_word[0, i, :].unsqueeze(0))
        
        if not all_standard_embeds_list or not all_projection_embeds_list:
            print("No embeddings generated for point cloud. Skipping visualization.")
            return

        all_standard_embeds = torch.cat(all_standard_embeds_list, dim=0).cpu().numpy()
        all_projection_embeds = torch.cat(all_projection_embeds_list, dim=0).cpu().numpy()

        if all_standard_embeds.shape[0] < 2 or all_projection_embeds.shape[0] < 2:
            print("Not enough data points for PCA. Skipping point cloud visualization.")
            return

        # PCA reduction
        pca_standard = PCA(n_components=2, random_state=42)
        standard_2d = pca_standard.fit_transform(all_standard_embeds)
        
        pca_projection = PCA(n_components=2, random_state=42)
        projection_2d = pca_projection.fit_transform(all_projection_embeds)
        
        # Plotting
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))
        fig.suptitle("Character Embedding Point Cloud (Projected Space Model - PCA Reduced)", fontsize=16)

        # Plot Standard d_model Embeddings
        axes[0].scatter(standard_2d[:, 0], standard_2d[:, 1], alpha=0.7)
        for i, label in enumerate(all_labels):
            axes[0].annotate(label, (standard_2d[i, 0], standard_2d[i, 1]), fontsize=8)
        axes[0].set_title(f'Standard Embeddings (d_model={self.projected_space_model.d_model})')
        axes[0].set_xlabel('PCA Component 1')
        axes[0].set_ylabel('PCA Component 2')
        axes[0].grid(True, linestyle='--', alpha=0.5)

        # Plot Projected d_projection Embeddings
        axes[1].scatter(projection_2d[:, 0], projection_2d[:, 1], alpha=0.7, color='orange')
        for i, label in enumerate(all_labels):
            axes[1].annotate(label, (projection_2d[i, 0], projection_2d[i, 1]), fontsize=8)
        axes[1].set_title(f'Projected Embeddings (d_projection={self.projected_space_model.d_projection})')
        axes[1].set_xlabel('PCA Component 1')
        axes[1].set_ylabel('PCA Component 2')
        axes[1].grid(True, linestyle='--', alpha=0.5)
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.savefig(filename, dpi=150)
        plt.close()
        print(f"Point cloud visualization saved to '{filename}'")

    def _generate_tokens(self, model, prompt, max_length, temperature, is_standard=False):
        input_ids = torch.tensor([self.tokenizer.encode(prompt)], device=self.device)
        
        model.eval()
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
        all_ngrams = set()
        for text in texts:
            for i in range(len(text) - n + 1):
                all_ngrams.add(text[i:i+n])
        return len(all_ngrams) / max(1, sum(len(text) - n + 1 for text in texts))
    
    def _forward_from_embeddings(self, model, embeddings, is_standard=False):
        seq_len = embeddings.size(1)
        current_device = embeddings.device
        
        pe_slice = model.positional_encoding[:, :seq_len, :].to(current_device)
        x = embeddings + pe_slice 
        x = model.dropout(x)
        
        mask = model.create_causal_mask(seq_len, current_device) 
        
        for block in model.transformer_blocks:
            x = block(x, mask)
        
        x = model.norm(x)
        
        if is_standard:
            return model.output_projection(x)
        else:
            return model.token_embedder.get_logits_from_projected_space(x)

    def _estimate_mutual_information(self, input_ids, hidden_repr):
        input_flat = input_ids.reshape(-1).cpu().numpy()
        hidden_flat = hidden_repr.mean(dim=-1).reshape(-1).cpu().detach().numpy()
        
        hidden_bins = np.histogram_bin_edges(hidden_flat, bins=10)
        hidden_discrete = np.digitize(hidden_flat, hidden_bins)
        
        joint_hist, _, _ = np.histogram2d(input_flat, hidden_discrete, bins=10)
        joint_hist = joint_hist / joint_hist.sum()
        
        marginal_input = joint_hist.sum(axis=1)
        marginal_hidden = joint_hist.sum(axis=0)
        
        h_input = -np.sum(marginal_input * np.log(marginal_input + 1e-8))
        h_hidden = -np.sum(marginal_hidden * np.log(marginal_hidden + 1e-8))
        h_joint = -np.sum(joint_hist * np.log(joint_hist + 1e-8))
        
        return h_input + h_hidden - h_joint

def calculate_perplexity(model, data_loader, criterion, device, is_standard_model=False, tokenizer=None):
    model.eval()
    total_loss = 0
    total_tokens = 0
    
    pad_token_id = 0
    if tokenizer and hasattr(tokenizer, 'pad_id') and tokenizer.pad_id is not None:
        pad_token_id = tokenizer.pad_id

    with torch.no_grad():
        for batch in data_loader:
            if isinstance(batch, (list, tuple)) and len(batch) == 2:
                input_ids, target_ids = batch
            else:
                input_ids = batch
                target_ids = input_ids[:, 1:].contiguous()
                input_ids = input_ids[:, :-1].contiguous()

            input_ids, target_ids = input_ids.to(device), target_ids.to(device)
            
            if is_standard_model:
                logits = model(input_ids)
            else:
                logits, _ = model(input_ids)
            
            loss = criterion(logits.view(-1, logits.size(-1)), target_ids.view(-1))

            active_tokens_in_batch = (target_ids.view(-1) != pad_token_id).sum().item()
            if active_tokens_in_batch > 0:
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
    model.train()
    return perplexity

def run_comprehensive_tests():
    """Run all tests"""
    print("Loading models and data for ProjectedSpace Concept Test...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataset_name = "wikitext"
    dataset_config = "wikitext-2-raw-v1"
    max_train_samples = 5000
    max_val_samples = 500
    train_max_length = 128
    eval_max_length = 128

    print(f"Loading dataset: {dataset_name} ({dataset_config})")
    raw_datasets = load_dataset(dataset_name, dataset_config)

    train_texts = [text for text in raw_datasets['train']['text'] if text.strip()][:max_train_samples]
    val_texts = [text for text in raw_datasets['validation']['text'] if text.strip()][:max_val_samples]

    if not train_texts:
        raise ValueError("No training texts loaded. Check dataset and filtering.")
    if not val_texts and max_val_samples > 0:
        print("Warning: No validation texts loaded. Using a subset of training data for validation.")
        val_texts = train_texts[:min(len(train_texts), max_val_samples if max_val_samples > 0 else 100)]
    elif not val_texts:
        print("Warning: No validation texts loaded and max_val_samples is 0. Perplexity will not be calculated.")

    tokenizer = CharTokenizer()
    print("Building tokenizer vocab on training data...")
    tokenizer.build_vocab(train_texts + val_texts)
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    
    train_dataset = TextDataset(train_texts, tokenizer, max_length=train_max_length)
    val_dataset = TextDataset(val_texts, tokenizer, max_length=eval_max_length)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2, pin_memory=True if device.type == 'cuda' else False)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2, pin_memory=True if device.type == 'cuda' else False) if val_texts else None
    
    d_model_val = 256
    d_projection_val = d_model_val

    model_params_projected = dict(
        vocab_size=tokenizer.vocab_size,
        d_model=d_model_val,
        d_projection=d_projection_val,
        n_heads=8,
        n_layers=4,
        d_ff_projection=d_projection_val * 4,
        max_seq_len=train_max_length,
        dropout=0.1
    )
    
    model_params_std = dict(
        vocab_size=tokenizer.vocab_size,
        d_model=d_model_val,
        n_heads=8,
        n_layers=4,
        d_ff=d_model_val * 4,
        max_seq_len=train_max_length,
        dropout=0.1
    )
    
    projected_space_model = ProjectedSpaceTransformer(**model_params_projected).to(device)
    standard_model = StandardTransformer(**model_params_std).to(device)
    
    epochs = 10
    lr = 1e-4
    projection_regularization_weight = 0.01 
    aux_relational_penalty = 0.001 # Example weight for auxiliary relational loss

    print(f"Training ProjectedSpace Model for {epochs} epochs...")
    train_projected_space_model(projected_space_model, train_loader, epochs=epochs, lr=lr, 
                           device=device, tokenizer=tokenizer, 
                           orth_loss_weight=projection_regularization_weight,
                           aux_relational_loss_weight=aux_relational_penalty)
    
    print(f"\nTraining Standard Model for {epochs} epochs...")
    optimizer = torch.optim.Adam(standard_model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=epochs//3, gamma=0.1)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.char_to_id.get('<PAD>', 0), reduction='mean')
    
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
            torch.nn.utils.clip_grad_norm_(standard_model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
            num_batches += 1
        scheduler.step()
        avg_epoch_loss = epoch_loss / num_batches
        print(f"Standard Model - Epoch {epoch+1}/{epochs}, Avg Loss: {avg_epoch_loss:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")
    
    perplexity_criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.char_to_id.get('<PAD>', 0), reduction='mean')

    ps_perplexity = float('inf')
    std_perplexity = float('inf')

    if val_loader:
        ps_perplexity = calculate_perplexity(projected_space_model, val_loader, perplexity_criterion, device, is_standard_model=False, tokenizer=tokenizer)
        std_perplexity = calculate_perplexity(standard_model, val_loader, perplexity_criterion, device, is_standard_model=True, tokenizer=tokenizer)
        print(f"ProjectedSpace Model Perplexity: {ps_perplexity:.2f}")
        print(f"Standard Model Perplexity: {std_perplexity:.2f}")
    else:
        print("Skipping perplexity calculation as no validation data/loader was available.")

    tester = ProjectedSpaceConceptTester(projected_space_model, standard_model, tokenizer)
    
    print("\nRunning comprehensive tests for ProjectedSpace model...")
    
    test_sequences = ["The quick brown fox", "Machine learning models", "Neural networks are", "Deep learning has"]
    independence_results = tester.test_embedding_independence(test_sequences)
    
    prompts = ["The ", "Machine learning ", "Neural network is ", "Deep learning can "]
    diversity_results = tester.test_prediction_diversity(prompts, num_samples=10)
    
    noise_levels = [0.0, 0.05, 0.1, 0.2, 0.5]
    robustness_results = tester.test_robustness_to_perturbation(
        "The quick brown fox jumps over the lazy dog", noise_levels
    )
    
    bottleneck_results = tester.test_information_bottleneck(test_sequences)
    
    all_results = {
        'meta_info': {
            'dataset_name': dataset_name,
            'dataset_config': dataset_config,
            'max_train_samples': max_train_samples,
            'max_val_samples': max_val_samples,
            'train_max_length': train_max_length,
            'epochs': epochs,
            'learning_rate': lr,
            'model_params_projected': model_params_projected,
            'model_params_standard': model_params_std,
            'projected_space_perplexity': ps_perplexity if val_loader else 'N/A',
            'standard_perplexity': std_perplexity if val_loader else 'N/A',
        },
        'independence': independence_results,
        'diversity': diversity_results,
        'robustness': robustness_results,
        'bottleneck': bottleneck_results
    }
    
    results_filename = 'projected_space_test_results.json'
    print(f"\nSaving detailed results to '{results_filename}'")
    with open(results_filename, 'w') as f:
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
    
    tester.visualize_results(all_results)

    # Define words for point cloud visualization
    point_cloud_words = [
        "king", "queen", "man", "woman", 
        "apple", "orange", "fruit", 
        "happy", "sad", "joy", "fear",
        "run", "walk", "fast", "slow"
    ]
    tester.visualize_embedding_point_cloud(point_cloud_words)
    
    print("\n" + "="*50)
    print("COMPREHENSIVE PROJECTED SPACE TEST SUMMARY")
    print("="*50)

    if val_loader:
        print(f"\nPERPLEXITY (on {dataset_name} validation - {max_val_samples} samples):")
        print(f"   ProjectedSpace Model: {ps_perplexity:.2f}")
        print(f"   Standard Model:   {std_perplexity:.2f}")
    
    print("\n1. EMBEDDING INDEPENDENCE (Avg. Off-Diagonal Covariance):")
    avg_ps_indep = np.nanmean(independence_results['projected_space_independence'])
    avg_dmodel_indep = np.nanmean(independence_results['d_model_pre_projection_independence'])
    print(f"   ProjectedSpace (d_projection space): {avg_ps_indep:.4f}")
    print(f"   MS Pre-Projection (d_model space): {avg_dmodel_indep:.4f}")
    if avg_dmodel_indep > 0 : print(f"   Reduction from d_model to d_projection: {(avg_dmodel_indep - avg_ps_indep) / avg_dmodel_indep * 100:.1f}%")
    
    print("\n2. PREDICTION DIVERSITY:")
    avg_ps_div = np.mean(diversity_results['projected_space_diversity'])
    avg_std_div = np.mean(diversity_results['standard_diversity'])
    print(f"   ProjectedSpace: {avg_ps_div:.4f}")
    print(f"   Standard:   {avg_std_div:.4f}")
    if avg_std_div > 0 : print(f"   Improvement: {(avg_ps_div - avg_std_div) / avg_std_div * 100:.1f}%")
    
    print("\n3. ROBUSTNESS (avg KL divergence):")
    avg_ps_rob = np.mean(robustness_results['projected_space_robustness'])
    avg_std_rob = np.mean(robustness_results['standard_robustness'])
    print(f"   ProjectedSpace: {avg_ps_rob:.4f}")
    print(f"   Standard:   {avg_std_rob:.4f}")
    if avg_std_rob > 0 : print(f"   Improvement (lower KL is better): {(avg_std_rob - avg_ps_rob) / avg_std_rob * 100:.1f}%")
    
    print("\n4. INFORMATION PROPERTIES:")
    avg_ps_ent = np.mean(bottleneck_results['projected_space_entropy'])
    avg_std_ent = np.mean(bottleneck_results['standard_entropy'])
    avg_ps_mi = np.nanmean(bottleneck_results['projected_space_mutual_info'])
    avg_std_mi = np.nanmean(bottleneck_results['standard_mutual_info'])
    
    avg_ps_mi = np.nan_to_num(avg_ps_mi, nan=0.0)
    avg_std_mi = np.nan_to_num(avg_std_mi, nan=0.0)

    print(f"   Entropy - ProjectedSpace: {avg_ps_ent:.4f}, Standard: {avg_std_ent:.4f}")
    print(f"   Mutual Info - ProjectedSpace: {avg_ps_mi:.4f}, Standard: {avg_std_mi:.4f}")
    
    print(f"\nAnalysis complete! Check 'projected_space_analysis.png' for visualizations.")
    print(f"Detailed results saved to '{results_filename}'")

if __name__ == "__main__":
    run_comprehensive_tests() 