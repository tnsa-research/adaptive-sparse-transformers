import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# 1. Positional Encoding (Standard Sinusoidal)
# This provides positional information to the input embeddings.
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_len=2048):
        super().__init__()
        self.d_model = d_model
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0) # Add batch dimension for broadcasting
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: (batch_size, seq_len, d_model)
        seq_len = x.shape[1]
        # Add positional encoding to input embeddings
        x = x + self.pe[:, :seq_len, :]
        return x

# Rotary Positional Embedding (Applied to Query and Key tensors)
# This is used within the attention mechanism and the attention gate.
class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len=2048):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.max_seq_len = max_seq_len

    def forward(self, x, seq_len=None):
        # x shape: (batch_size, num_heads, seq_len, head_dim) or (batch_size * num_heads, block_seq_len, head_dim)
        if seq_len is None:
            seq_len = x.shape[-2]
        
        # Ensure t has the correct length for the current sequence
        t = torch.arange(seq_len, device=x.device, dtype=self.inv_freq.dtype)
        
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1) # (seq_len, dim)
        
        # Reshape emb to match x's dimensions for broadcasting
        if x.dim() == 4: # For Multi-Head Attention Q/K
            cos_emb = emb.cos().unsqueeze(0).unsqueeze(0)
            sin_emb = emb.sin().unsqueeze(0).unsqueeze(0)
        else: # x.dim() == 3 (for AttentionGate Q/K)
            cos_emb = emb.cos().unsqueeze(0)
            sin_emb = emb.sin().unsqueeze(0)

        return x * cos_emb + self._rotate_half(x) * sin_emb

    def _rotate_half(self, x):
        x1, x2 = x[..., :x.shape[-1] // 2], x[..., x.shape[-1] // 2:]
        return torch.cat((-x2, x1), dim=-1)

# 2. AttentionGate
# This module learns to identify significant blocks in the attention map.
# Inspired by SeerAttention, it pools Q and K, applies linear layers, and computes gating scores. [2, 3]
class AttentionGate(nn.Module):
    def __init__(self, d_model, num_heads, block_size, max_seq_len=2048):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.block_size = block_size
        self.head_dim = d_model // num_heads
        
        # Calculate maximum block sequence length for RoPE initialization
        self.max_block_seq_len = max_seq_len // block_size
        assert max_seq_len % block_size == 0, "max_seq_len must be divisible by block_size"

        # Pooling layers for Q and K [2]
        # SeerAttention suggests average pooling for Q, and a combination of max and min pooling for K.
        # Here, we use average and max pooling for K for simplicity.
        self.q_pool = nn.AvgPool1d(kernel_size=block_size, stride=block_size)
        self.k_max_pool = nn.MaxPool1d(kernel_size=block_size, stride=block_size)
        self.k_avg_pool = nn.AvgPool1d(kernel_size=block_size, stride=block_size)

        # Linear layers for the gate [2]
        self.gate_q_proj = nn.Linear(self.head_dim, self.head_dim)
        self.gate_k_proj = nn.Linear(self.head_dim * 2, self.head_dim) # K has 2 pooled versions (max+avg)

        # Separate RoPE for the gate, applied at block level [2]
        self.gate_rope = RotaryPositionalEmbedding(self.head_dim, self.max_block_seq_len)

    def forward(self, Q, K):
        # Q, K shape: (batch_size, num_heads, seq_len, head_dim)
        batch_size, num_heads, seq_len, head_dim = Q.shape

        # Reshape for pooling: (batch_size * num_heads, head_dim, seq_len)
        Q_pooled_input = Q.view(batch_size * num_heads, seq_len, head_dim).permute(0, 2, 1)
        K_pooled_input = K.view(batch_size * num_heads, seq_len, head_dim).permute(0, 2, 1)

        # Apply pooling [2]
        q_pooled = self.q_pool(Q_pooled_input).permute(0, 2, 1) # (B*H, current_block_seq_len, head_dim)
        k_max_pooled = self.k_max_pool(K_pooled_input).permute(0, 2, 1)
        k_avg_pooled = self.k_avg_pool(K_pooled_input).permute(0, 2, 1)
        k_pooled = torch.cat([k_max_pooled, k_avg_pooled], dim=-1) # Concatenate max and avg pooled K [2]

        # Apply linear layers [2]
        q_gate = self.gate_q_proj(q_pooled)
        k_gate = self.gate_k_proj(k_pooled)

        # Apply RoPE to pooled Q and K for the gate [2]
        current_block_seq_len = q_gate.shape[-2]
        q_gate = self.gate_rope(q_gate, seq_len=current_block_seq_len)
        k_gate = self.gate_rope(k_gate, seq_len=current_block_seq_len)

        # Compute gating scores (similar to attention scores) [2]
        gating_scores = torch.matmul(q_gate, k_gate.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Apply sigmoid to get importance scores [3]
        gating_scores = torch.sigmoid(gating_scores) # (batch_size * num_heads, current_block_seq_len, current_block_seq_len)

        # Reshape back to (batch_size, num_heads, current_block_seq_len, current_block_seq_len)
        gating_scores = gating_scores.view(batch_size, num_heads, current_block_seq_len, current_block_seq_len)
        
        return gating_scores

# 3. Adaptive Sparse Multi-Head Attention
# This module replaces standard Multi-Head Attention, incorporating the AttentionGate
# to dynamically determine sparse patterns. It also supports a hybrid approach with dense heads.
class AdaptiveSparseMultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, block_size, sparsity_threshold, num_dense_heads=0, max_seq_len=2048):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.block_size = block_size
        self.sparsity_threshold = sparsity_threshold
        self.num_dense_heads = num_dense_heads
        
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        assert num_dense_heads <= num_heads, "num_dense_heads cannot exceed total num_heads"

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.rope = RotaryPositionalEmbedding(self.head_dim, max_seq_len)
        
        # Initialize AttentionGate only if there are sparse heads
        num_sparse_heads = self.num_heads - self.num_dense_heads
        if num_sparse_heads > 0:
            self.attention_gate = AttentionGate(d_model, num_sparse_heads, block_size, max_seq_len)
        else:
            self.attention_gate = None

    def forward(self, query, key, value, mask=None):
        batch_size, seq_len, d_model = query.shape

        # Project to Q, K, V and reshape for multi-head attention
        Q = self.q_proj(query).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(key).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(value).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE to Q and K
        Q = self.rope(Q, seq_len=seq_len)
        K = self.rope(K, seq_len=seq_len)

        all_head_outputs =

        # --- Process Dense Heads ---
        if self.num_dense_heads > 0:
            Q_dense = Q[:, :self.num_dense_heads]
            K_dense = K[:, :self.num_dense_heads]
            V_dense = V[:, :self.num_dense_heads]

            scores_dense = torch.matmul(Q_dense, K_dense.transpose(-2, -1)) / math.sqrt(self.head_dim)
            if mask is not None:
                # Apply causal/padding mask to dense heads
                scores_dense = scores_dense.masked_fill(mask == float('-inf'), float('-inf'))
            attention_weights_dense = F.softmax(scores_dense, dim=-1)
            output_dense = torch.matmul(attention_weights_dense, V_dense)
            all_head_outputs.append(output_dense)

        # --- Process Sparse Heads ---
        num_sparse_heads = self.num_heads - self.num_dense_heads
        if num_sparse_heads > 0:
            Q_sparse = Q[:, self.num_dense_heads:]
            K_sparse = K[:, self.num_dense_heads:]
            V_sparse = V[:, self.num_dense_heads:]

            # Get block-level gating scores from AttentionGate [2, 3]
            # Pass only the sparse heads' Q and K to the gate
            gating_scores = self.attention_gate(Q_sparse, K_sparse) # (B, num_sparse_heads, block_seq_len, block_seq_len)
            
            # Create a block-level binary mask from gating scores [3]
            block_mask = (gating_scores > self.sparsity_threshold).float()

            # Expand block mask to full sequence length mask
            # NOTE: This expansion is for conceptual demonstration.
            # In a real, hardware-optimized sparse attention implementation (e.g., FlashAttention with sparsity),
            # this full mask would not be materialized explicitly to save memory and computation.
            # Instead, specialized kernels would directly compute attention only for the active blocks.
            current_block_seq_len = seq_len // self.block_size
            full_sparse_mask = torch.zeros(batch_size, num_sparse_heads, seq_len, seq_len, device=query.device)
            for b in range(batch_size):
                for h in range(num_sparse_heads):
                    for i in range(current_block_seq_len):
                        for j in range(current_block_seq_len):
                            if block_mask[b, h, i, j] == 1:
                                full_sparse_mask[b, h, 
                                                 i*self.block_size:(i+1)*self.block_size, 
                                                 j*self.block_size:(j+1)*self.block_size] = 1.0

            scores_sparse = torch.matmul(Q_sparse, K_sparse.transpose(-2, -1)) / math.sqrt(self.head_dim)
            
            # Apply the full sparse mask
            scores_sparse = scores_sparse.masked_fill(full_sparse_mask == 0, float('-inf'))
            if mask is not None: # Apply causal/padding mask if present
                scores_sparse = scores_sparse.masked_fill(mask == float('-inf'), float('-inf'))

            attention_weights_sparse = F.softmax(scores_sparse, dim=-1)
            output_sparse = torch.matmul(attention_weights_sparse, V_sparse)
            all_head_outputs.append(output_sparse)

        # Concatenate all head outputs (dense and sparse)
        output = torch.cat(all_head_outputs, dim=1)

        # Reshape back to original (batch_size, seq_len, d_model) and apply final projection
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        return self.out_proj(output)

# 4. Adaptive Sparse Transformer Block
# This is a standard Transformer block structure, but with our custom attention mechanism.
class AdaptiveSparseTransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, ffn_hidden_size, dropout_rate, block_size, sparsity_threshold, num_dense_heads=0, max_seq_len=2048):
        super().__init__()
        self.attention = AdaptiveSparseMultiHeadAttention(d_model, num_heads, block_size, sparsity_threshold, num_dense_heads, max_seq_len)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout_rate)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_hidden_size),
            nn.ReLU(), # Standard ReLU activation
            nn.Linear(ffn_hidden_size, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout_rate)

    def forward(self, x, mask=None):
        # Attention sub-layer with residual connection and layer norm [4]
        attn_output = self.attention(x, x, x, mask)
        x = x + self.dropout1(attn_output)
        x = self.norm1(x)

        # Feed-Forward Network sub-layer with residual connection and layer norm [4]
        ffn_output = self.ffn(x)
        x = x + self.dropout2(ffn_output)
        x = self.norm2(x)
        return x

# 5. Simplified LLM (targeting ~7M parameters)
# This model stacks multiple AdaptiveSparseTransformerBlock instances.
class SparseLLM_7M(nn.Module):
    def __init__(self, vocab_size, d_model, num_layers, num_heads, ffn_hidden_size, block_size, sparsity_threshold, num_dense_heads=0, max_seq_len=2048, dropout_rate=0.1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_len)
        self.dropout = nn.Dropout(dropout_rate)

        self.layers = nn.ModuleList()
        self.norm_final = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids, attention_mask=None):
        seq_len = input_ids.shape[1]
        x = self.embedding(input_ids)
        x = self.positional_encoding(x)
        x = self.dropout(x)

        # Create causal mask for decoder-like behavior (prevents attending to future tokens) [4]
        # This mask will have -inf for future positions.
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=input_ids.device, dtype=torch.bool), diagonal=1)
        causal_mask_float = torch.zeros(seq_len, seq_len, device=input_ids.device, dtype=torch.float)
        causal_mask_float.masked_fill_(causal_mask, float('-inf')) # Fill future positions with -inf

        # Combine with padding attention_mask if provided
        # padding_mask: (batch_size, seq_len) -> (batch_size, 1, 1, seq_len) for broadcasting
        if attention_mask is not None:
            padding_mask = (attention_mask == 0).unsqueeze(1).unsqueeze(2) # True for padded positions
            # Any position that is either future OR padded should be masked
            # Adding -inf to 0.0 results in -inf, adding -inf to -inf results in -inf
            combined_mask = causal_mask_float.unsqueeze(0) + padding_mask.float().masked_fill(padding_mask, float('-inf'))
        else:
            combined_mask = causal_mask_float.unsqueeze(0) # Add batch dim for consistency

        for layer in self.layers:
            x = layer(x, mask=combined_mask)

        x = self.norm_final(x)
        logits = self.lm_head(x)
        return logits

# Helper function to count trainable parameters in a model
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# --- Example Usage and Parameter Calculation for a ~7M LLM ---
# These parameters are chosen to approximate a 7M parameter model.
# The exact count may vary slightly due to biases in linear layers and LayerNorms.
VOCAB_SIZE = 10000       # Size of the vocabulary
D_MODEL = 192            # Dimensionality of the model embeddings and hidden states
NUM_LAYERS = 8           # Number of Transformer blocks
NUM_HEADS = 3            # Number of attention heads
FFN_HIDDEN_SIZE = D_MODEL * 4 # Hidden size of the Feed-Forward Network (typically 4x d_model)
BLOCK_SIZE = 64          # Block size for the AttentionGate's pooling [2]
SPARSITY_THRESHOLD = 0.5 # Threshold for activating attention blocks (0.0 to 1.0) [3]
NUM_DENSE_HEADS = 1      # Number of attention heads that will always compute full attention [5]
MAX_SEQ_LEN = 1024       # Maximum sequence length the model is designed to handle

# Instantiate the LLM
model = SparseLLM_7M(
    vocab_size=VOCAB_SIZE,
    d_model=D_MODEL,
    num_layers=NUM_LAYERS,
    num_heads=NUM_HEADS,
    ffn_hidden_size=FFN_HIDDEN_SIZE,
    block_size=BLOCK_SIZE,
    sparsity_threshold=SPARSITY_THRESHOLD,
    num_dense_heads=NUM_DENSE_HEADS,
    max_seq_len=MAX_SEQ_LEN
)

# Print total parameters
total_params = count_parameters(model)
print(f"Total trainable parameters in SparseLLM_7M: {total_params / 1e6:.2f} Million")

# Example forward pass (uncomment to test)
# input_ids = torch.randint(0, VOCAB_SIZE, (2, 512)) # Batch size 2, sequence length 512
# attention_mask = torch.ones(2, 512, dtype=torch.long) # All tokens valid
# output_logits = model(input_ids, attention_mask)
# print(f"Output logits shape: {output_logits.shape}")
