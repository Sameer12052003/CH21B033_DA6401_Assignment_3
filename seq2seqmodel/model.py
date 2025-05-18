import torch
import torch.nn as nn
import torch.nn.functional as F

# Encoder
class Encoder(nn.Module):
    def __init__(self, config,source_vocab_size):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(source_vocab_size, config['embedding_dim'])

        rnn_cell = getattr(nn, config["cell_type"])
        self.rnn = rnn_cell(
            input_size=config['embedding_dim'],
            hidden_size=config['hidden_size'],
            num_layers=config['num_layers'],
            dropout=config['dropout'] if config['num_layers'] > 1 else 0,
            batch_first=True
        )

    def forward(self, x):
        embedded = self.embedding(x)
        outputs, hidden = self.rnn(embedded)
        return outputs, hidden
    
# Luong Attention
class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.attn = nn.Linear(hidden_size * 2, hidden_size)
        self.v = nn.Parameter(torch.rand(hidden_size))

    def forward(self, hidden, encoder_outputs):
        # hidden: [B, H], encoder_outputs: [B, S, H]
        batch_size, seq_len, hidden_size = encoder_outputs.size()

        hidden = hidden.unsqueeze(1).repeat(1, seq_len, 1)  # [B, S, H]
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))  # [B, S, H]
        energy = energy.transpose(1, 2)  # [B, H, S]
        v = self.v.repeat(batch_size, 1).unsqueeze(1)  # [B, 1, H]
        attn_weights = torch.bmm(v, energy).squeeze(1)  # [B, S]
        return F.softmax(attn_weights, dim=1)  # [B, S]

# Decoder
class Decoder(nn.Module):
    def __init__(self, config,target_vocab_size, attn_mechanism=False):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(target_vocab_size, config['embedding_dim'])
        self.attn_mechanism = attn_mechanism

        rnn_cell = getattr(nn, config["cell_type"])
        self.rnn = rnn_cell(
            input_size=config['embedding_dim'],
            hidden_size=config['hidden_size'],
            num_layers=config['num_layers'],
            dropout=config['dropout'] if config['num_layers'] > 1 else 0,
            batch_first=True
        )
        
        self.attention = Attention(config['hidden_size'])
        self.out = nn.Linear(config['hidden_size'], target_vocab_size)

    def forward(self, x, hidden,encoder_outputs):
        x = x.unsqueeze(1)  # (B) -> (B, 1)
        embedded = self.embedding(x)  # (B, 1, E)
        
        if self.attn_mechanism:
            # Get attention weights using last layer's hidden state
            attn_weights = self.attention(hidden[-1], encoder_outputs)  # [B, S]
            context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs)  # [B, 1, H]

            rnn_input = torch.cat((embedded, context), dim=2)  # [B, 1, E+H]
            output, hidden = self.rnn(rnn_input, hidden)
            output = output.squeeze(1)  # [B, H]
            context = context.squeeze(1)  # [B, H]

            prediction = self.out(torch.cat((output, context), dim=1))  # [B, V]
            return prediction, hidden
        
        else:
            output, hidden = self.rnn(embedded, hidden)
            output = self.out(output.squeeze(1))  # (B, V)
            return output, hidden

# Seq2Seq Model
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, config, target_vocab_size):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.config = config
        self.target_vocab_size = target_vocab_size

    def forward(self, src, tgt, teacher_forcing_ratio=0.5):
        batch_size, tgt_len = tgt.size()
        outputs = torch.zeros(batch_size, tgt_len, self.target_vocab_size).to(src.device)

        encoder_outputs, hidden = self.encoder(src)

        # Prepare initial input (start-of-sequence token)
        input = tgt[:, 0]

        for t in range(1, tgt_len):
            output, hidden = self.decoder(input, hidden,encoder_outputs)
            outputs[:, t] = output
            top1 = output.argmax(1)

            use_teacher_forcing = torch.rand(1).item() < teacher_forcing_ratio
            input = tgt[:, t] if use_teacher_forcing else top1

        return outputs
