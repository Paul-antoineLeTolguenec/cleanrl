import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, input_dim, num_heads, attention_dim):
        super(MultiHeadAttention, self).__init__()
        assert attention_dim % num_heads == 0, "attention_dim doit être divisible par num_heads"
        
        self.num_heads = num_heads
        self.attention_dim_per_head = attention_dim // num_heads
        
        self.query = nn.Linear(input_dim, attention_dim)
        self.key = nn.Linear(input_dim, attention_dim)
        self.value = nn.Linear(input_dim, attention_dim)
        
        self.final_linear = nn.Linear(attention_dim, input_dim)
        
    def forward(self, x):
        batch_size, seq_length, _ = x.size()

        # Transformation des entrées en requêtes, clés et valeurs
        Q = self.query(x).view(batch_size, seq_length, self.num_heads, self.attention_dim_per_head)
        K = self.key(x).view(batch_size, seq_length, self.num_heads, self.attention_dim_per_head)
        V = self.value(x).view(batch_size, seq_length, self.num_heads, self.attention_dim_per_head)

        # Transposition pour obtenir la dimension des têtes en avant
        Q = Q.transpose(1, 2)  # (batch_size, num_heads, seq_length, attention_dim_per_head)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        # Calcul des scores d'attention avec scaling
        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.attention_dim_per_head, dtype=torch.float32))
        attention_weights = F.softmax(attention_scores, dim=-1)

        # Application des scores d'attention aux valeurs
        weighted_values = torch.matmul(attention_weights, V)  # (batch_size, num_heads, seq_length, attention_dim_per_head)
        weighted_values = weighted_values.transpose(1, 2).contiguous().view(batch_size, seq_length, -1)

        # Appliquer une dernière couche linéaire
        output = self.final_linear(weighted_values)

        return output, attention_weights

# Exemple d'utilisation
input_dim = 64
num_heads = 8
attention_dim = 128  # Doit être un multiple du nombre de têtes

x = torch.rand(10, 20, input_dim)  # (batch_size, seq_length, input_dim)
attention_layer = MultiHeadAttention(input_dim, num_heads, attention_dim)
output, attention_weights = attention_layer(x)

print("Output Shape:", output.shape)  # (batch_size, seq_length, input_dim)
# print("Attention Weights Shape:", attention_weights.shape)  # (batch_size, num_heads, seq_length, seq_length)
