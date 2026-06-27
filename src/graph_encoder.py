import mlx.core as mx
import mlx.nn as nn
import math

class DenseGATLayer(nn.Module):
    """
    Graph Attention Network layer using dense adjacency matrices for MLX.
    Optimized for small graphs (e.g., reasoning traces where N < 100).
    """
    def __init__(self, in_features: int, out_features: int, heads: int = 4, concat: bool = True, dropout: float = 0.1):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.heads = heads
        self.concat = concat
        self.dropout = dropout

        # Linear projection for node features
        self.W = nn.Linear(in_features, heads * out_features, bias=False)
        
        # Attention mechanisms for source and destination nodes
        self.a_src = nn.Linear(out_features, 1, bias=False)
        self.a_dst = nn.Linear(out_features, 1, bias=False)

    def __call__(self, x: mx.array, adj_mask: mx.array) -> mx.array:
        """
        x: (B, N, in_features) - Node features
        adj_mask: (B, N, N) - Adjacency matrix where 1 means edge exists, 0 means no edge
        """
        B, N, _ = x.shape
        
        # 1. Project features
        # x_proj shape: (B, N, heads, out_features)
        x_proj = self.W(x).reshape(B, N, self.heads, self.out_features)
        
        # 2. Compute attention scores
        # src_scores shape: (B, N, heads, 1)
        src_scores = self.a_src(x_proj)
        dst_scores = self.a_dst(x_proj)
        
        # 3. Combine scores for edges
        src_expanded = mx.expand_dims(mx.squeeze(src_scores, -1), 2)  # (B, N, 1, heads)
        dst_expanded = mx.expand_dims(mx.squeeze(dst_scores, -1), 1)  # (B, 1, N, heads)
        
        # e shape: (B, N, N, heads)
        e = nn.leaky_relu(src_expanded + dst_expanded, negative_slope=0.2)
        
        # 4. Mask out non-edges
        mask = mx.expand_dims(adj_mask, -1)
        zero_vec = mx.full(e.shape, -1e9)
        attention = mx.where(mask > 0, e, zero_vec)
        
        # 5. Softmax over neighbors (axis 2 is destination nodes)
        attention = mx.softmax(attention, axis=2)
        
        # 6. Optional dropout (if training)
        if self.dropout > 0 and self.training:
            attention = nn.Dropout(self.dropout)(attention)
            
        # 7. Weighted sum of neighbor features
        h_prime = mx.sum(
            mx.expand_dims(attention, -1) * mx.expand_dims(x_proj, 1), 
            axis=2
        ) # Result: (B, N, heads, out_features)
        
        # 8. Concat or mean
        if self.concat:
            return h_prime.reshape(B, N, self.heads * self.out_features)
        else:
            return mx.mean(h_prime, axis=2)


class GraphAwareBridge(nn.Module):
    """
    Bridge module that processes structured reasoning traces (JSON graphs) 
    into the sequence representations required by HRM-MLX.
    
    Now accepts both operation type IDs AND numerical argument values.
    """
    def __init__(self, vocab_size: int, d_model: int, num_value_features: int = 2, gat_hidden: int = 128, gat_layers: int = 3, heads: int = 4):
        super().__init__()
        
        # Embedding layer for operation types
        self.node_embedding = nn.Embedding(vocab_size, d_model - num_value_features)
        
        # Linear projection to fuse the concatenated [op_embed || values] into d_model
        self.value_proj = nn.Linear(d_model, d_model)
        
        # GAT layers
        self.layers = []
        
        # Input to hidden
        self.layers.append(DenseGATLayer(d_model, gat_hidden, heads=heads, concat=True))
        
        # Hidden to hidden
        for _ in range(gat_layers - 2):
            self.layers.append(DenseGATLayer(gat_hidden * heads, gat_hidden, heads=heads, concat=True))
            
        # Hidden to out
        self.layers.append(DenseGATLayer(gat_hidden * heads, d_model, heads=1, concat=False))

    def __call__(self, node_ids: mx.array, node_values: mx.array, adj_mask: mx.array) -> mx.array:
        """
        node_ids: (B, N) - Integer tokens representing operations
        node_values: (B, N, 2) - Numerical argument values [arg1, arg2] per node
        adj_mask: (B, N, N) - Adjacency matrix for dependency graph
        
        Returns:
        (B, N, d_model) - Embeddings compatible with HRM-MLX
        """
        # Embed operation type tokens: (B, N, d_model - 2)
        op_embed = self.node_embedding(node_ids)
        
        # Concatenate op embeddings with numerical values: (B, N, d_model)
        x = mx.concatenate([op_embed, node_values], axis=-1)
        
        # Project the fused features
        x = self.value_proj(x)
        
        # Pass through GAT layers
        for i, layer in enumerate(self.layers):
            x = layer(x, adj_mask)
            if i < len(self.layers) - 1:
                x = nn.elu(x)
                
        return x
