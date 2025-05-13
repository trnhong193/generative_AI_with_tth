import torch
import torch.nn as nn
import math

class InputEmbedding(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)
        # create a table to search embbedings with vocab size rows and each embedding has size: d_model (columns)

    def forward(self,x):
        """
        Forward pass to convert input token indices into dense, scaled embeddings.

        Args:
            x (torch.Tensor): Input tensor of token indices with shape (batch_size, sequence_length).

        Returns:
            torch.Tensor: Tensor of shape (batch_size, sequence_length, d_model)
                          containing the scaled embeddings.
        """
        x = self.embedding(x) * math.sqrt(self.d_model)
        
    
class PositionalEncoding(nn.Module):
    """
    Positional Encoding module to add positional information to the input embeddings.
    as transfomer dont have sequence structure as RNN
    """
    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        super().__init__()
        self.d_model  = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        # create a matrix of shape (seq_len, d_model) 
        pe = torch.zeros(seq_len, d_model)
        # create a vector of shape (seq_len, 1)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -math.log(10000.0) / d_model)    
        # apply sin, cos to odd and even positions
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, seq_len, d_model)

        self.register_buffer('pe', pe)  # register as buffer - constant in model, not a model parameter

    def forward(self, x):
            x = x + self.pe[:, :x.size(1), :].detach()
            return self.dropout(x)

class LayerNormalization(nn.Module):
    """
    Layer Normalization module to normalize the input tensor.
    chuẩn hóa input theo từng vector embedding - theo trục d_model
    """
    def __init__(self, eps = 10**-6) ->  None:
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1)) # Multiplied
        self.bias = nn.Parameter(torch.zeros(1)) #Added

    def forward(self, x):
        mean = x.mean(dim = -1, keepdim=True) # final dim: d_model
        std = x.std(dim = -1, keepdim=True) # braodcast: expand dim of tensor -> x - mean 
        x = (x - mean) / (std + self.eps)
        return self.alpha * x + self.bias
    
class FeedForwardBlock(nn.Module):
    """
    Một mạng FC gồm 2 lớp nằm giữa các lớp attention
    Linear(d_model → d_ff) → ReLU → Dropout → Linear(d_ff → d_model).
    """
    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None: 
        # d_ff: dim of hidden layer, d_model: dim of input
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff) # W1 and B1 -> d_model -> d_ff
        # size  of pe: (1, seq_len, d_model)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)  # W2 and B2
        # self.layer_norm = LayerNormalization() # layer norm
  
    def forward(self, x):
        # (Batch, seq_len, d_model) -> (Batch, seq_len, d_ff) -> (Batch, seq_len, d_model)
        # Batch la so luong sentence??
        # x = self.layer_norm(x)
        x = self.linear1(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x
    
class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, d_model: int, h: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.h = h # number of heads 
        assert d_model % h == 0, "d_model must be divisible by h"

        self.d_k = d_model // h 
        # self.d_v = d_model // h
        self.w_q = nn.Linear(d_model, d_model) # Wq
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)

        self.w_o = nn.Linear(d_model, d_model) # Wo
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        d_k = query.size(-1)
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None: # loai bo nhung vi tri khong can attention
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)
        attention_scores = torch.softmax(attention_scores, dim=-1)
        if dropout is not None:
            attention_scores = dropout(attention_scores) # (Batch, h, seq_len, seq_len)
        # (Batch, h, seq_len, d_k) @ (Batch, h, d_k, seq_len) -> (Batch, h, seq_len, seq_len)
        # (Batch, h, seq_len, seq_len) @ (Batch, h, seq_len, d_k) -> (Batch, h, seq_len, d_k)

        return (attention_scores @ value), attention_scores

    def forward(self, q, k, v, mask):
        query = self.w_q(q) # (Batch, seq_len, d_model) -> (Batch, seq_len, d_model)
        key = self.w_k(k)
        value = self.w_v(v)
        # (Batch, seq_len, d_model) -> (Batch, seq_len, h, d_k) -> (Batch, h, seq_len, d_k)
        query = query.view(query.size(0), query.size(1), self.h, self.d_k).transpose(1, 2)
        key = key.view(key.size(0), key.size(1), self.h, self.d_k).transpose(1, 2)
        value = value.view(value.size(0), value.size(1), self.h, self.d_k).transpose(1, 2)
        
        x, self.attention_scores = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)
        # (Batch, h, seq_len, d_k) -> (Batch, seq_len, h, d_k) -> (Batch, seq_len, d_model) 
        x = x.transpose(1,2).contiguous().view(x.size(0), x.size(1), self.d_model)
        # (Batch, seq_len, h, d_k) -> (Batch, seq_len, d_model)
        return self.w_o(x) # (Batch, seq_len, d_model)

class ResidualConnection(nn.Module):
    def __init__(self, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = LayerNormalization()

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.layer_norm(x)))

class EncoderBlock(nn.Module):
    def __init__(self, self_attention_block: MultiHeadAttentionBlock,feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connection = nn.ModuleList([ResidualConnection(dropout) for _ in range(2)]) # 2 residual connections
        # self.residual_connection1 = ResidualConnection(dropout)
        # self.residual_connection2 = ResidualConnection(dropout)

    def forward(self, x, src_mask):
        x = self.residual_connection[0](x, lambda x: self.self_attention_block(x, x, x, src_mask))
        x = self.residual_connection[1](x, self.feed_forward_block)
        return x

class Encoder(nn.Module):
    def __init__(self, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()
    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)

class DecoderBlock(nn.Module):
    def __init__(self, self_attention_block:MultiHeadAttentionBlock, cross_attention_block: MultiHeadAttentionBlock, feed_forward_block: FeedForwardBlock, dropout: float) -> None:
        super().__init__()
        self.self_attention_block = self_attention_block
        self.cross_attention_block = cross_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connection = nn.ModuleList([ResidualConnection(dropout) for _ in range(3)])

    def  forward(self, x, encoder_output, src_mask, tgt_mask):
        x = self.residual_connection[0](x, lambda x: self.self_attention_block(x, x, x, tgt_mask))
        x = self.residual_connection[1](x, lambda x: self.cross_attention_block(x, encoder_output, encoder_output, src_mask))
        x = self.residual_connection[2](x, self.feed_forward_block)
        return x 

class Decoder(nn.Module):
    def __init__(self, layers: nn.ModuleList):
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()
    def forward(self, x, encoder_output, src_mask, tgt_mask):
        for layer in self.layers: # layer - DecoderBlock
            x = layer(x, encoder_output, src_mask, tgt_mask)
        return self.norm(x)     

class ProjectionLayer(nn.Module):
    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.linear = nn.Linear(d_model, vocab_size)   # prj 
        self.softmax = nn.LogSoftmax(dim=-1) # dim = -1: last dim
    def forward(self, x):
        # (Batch, seq_len, d_model) -> (Batch, seq_len, vocab_size)
        x = self.linear(x)
        return self.softmax(x)
    

class Transfomer(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder, src_emb: InputEmbedding, tgt_emb: InputEmbedding, projection_layer: ProjectionLayer, src_pos: PositionalEncoding, tgt_pos: PositionalEncoding)
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_emb = src_emb
        self.tgt_emb = tgt_emb
        self.projection_layer = projection_layer
        self.src_pos = src_pos
        self.tgt_pos = tgt_pos