GELU = 8
SOFTMAX = 5 
LAYERNORM = 5

def flops_matmul(m, n, p):
    return 2 * m * n * p

def flops_attn(hidden_size, seq_len, n_heads):
    flops = 0
    flops += LAYERNORM  * hidden_size * seq_len # layer norm
    flops += flops_matmul(seq_len, hidden_size, 3 * hidden_size) # QKV
    flops += flops_matmul(seq_len, hidden_size, seq_len)  # QK^T
#    flops += seq_len * seq_len * n_heads # scaling
#    flops += SOFTMAX * seq_len * seq_len * n_heads  # softmax
    flops += flops_matmul(seq_len, seq_len, hidden_size) # AV
    flops += flops_matmul(seq_len, hidden_size, hidden_size) # fc out
    flops += seq_len * hidden_size # residual
    return flops

def flops_attn_fused(hidden_size, seq_len, n_heads, chunk_size, n_timesteps):
    n_blocks = max(1, seq_len // chunk_size)
    spatial_dim = seq_len // n_timesteps

    flops = 0
    flops += LAYERNORM  * hidden_size * seq_len # layer norm
    flops += flops_matmul(seq_len, hidden_size, 3 * hidden_size) # QKV
    flops += flops_attn_matmul(hidden_size, min(seq_len, chunk_size)) * n_blocks
    flops += flops_attn_matmul(hidden_size, n_timesteps) * spatial_dim
    flops += flops_matmul(seq_len, hidden_size, hidden_size) # fc out
    flops += seq_len * hidden_size # residual
    return flops

def flops_attn_matmul(hidden_size, seq_len):
    flops = 0
    flops += flops_matmul(seq_len, hidden_size, seq_len)  # QK^T
    flops += flops_matmul(seq_len, seq_len, hidden_size) # AV
    return flops
    

def flops_mlp(hidden_size, seq_len):
    flops = 0
    flops += LAYERNORM  * hidden_size * seq_len # layer norm
    flops += flops_matmul(seq_len, hidden_size, 4 * hidden_size)
    flops += GELU * 4 * hidden_size * seq_len
    flops += flops_matmul(seq_len, 4 * hidden_size, hidden_size)
    flops += seq_len * hidden_size # residual
    return flops


def flops_block(hidden_size, seq_len, n_heads):
    flops = 0
    flops += flops_attn(hidden_size, seq_len, n_heads)
    flops += flops_mlp(hidden_size, seq_len)
    return flops


def flops_tfm(hidden_size, seq_len, n_heads, num_layers):
    return flops_block(hidden_size, seq_len, n_heads) * num_layers


def flops_block2(hidden_size, seq_len, n_heads, chunk_size, n_timesteps):
    flops = 0
    n_blocks = max(1, seq_len // chunk_size)
    spatial_dim = seq_len // n_timesteps

#    flops += flops_attn(hidden_size, min(seq_len, chunk_size), n_heads) * n_blocks # full / chunked attn
#    flops += flops_attn(hidden_size, n_timesteps, n_heads) * spatial_dim # temporal axial attn
    flops += flops_attn_fused(hidden_size, seq_len, n_heads, chunk_size, n_timesteps)
    flops += flops_mlp(hidden_size, seq_len) # MLP
    return flops

def flops_tfm2(hidden_size, seq_len, n_heads, chunk_size, n_timesteps, num_layers):
    return flops_block2(hidden_size, seq_len, n_heads, chunk_size, n_timesteps) * num_layers

