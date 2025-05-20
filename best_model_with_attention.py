import os
from datetime import datetime
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import csv
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import os
import wandb
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import torch
import matplotlib
from matplotlib import cm
from matplotlib.colors import to_hex
from IPython.core.display import display, HTML
from data_preprocessing.data_preprocess import exact_match_accuracy, greedy_decode
from data_preprocessing.data_preprocess import (
    build_vocab, DakshinaCharDataset, collate_fn, load_tsv_data
)
from seq2seqmodel.model import Encoder, Decoder, Seq2Seq

# %% Best set of hyperparams
config = {
        'embedding_dim': 128,
        'hidden_size':  256,
        'num_layers': 3,
        'dropout':  0.3,
        'cell_type' :  'GRU',
        'beam_search' : 1,
        'epochs': 15,
        'max_len':  30,
    }

np.random.seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load and tokenize data

train_latin, train_devnag = load_tsv_data('/kaggle/input/dakshina-dataset-v1-0/dakshina_dataset_v1.0/hi/lexicons/hi.translit.sampled.train.tsv')
test_latin, test_devnag = load_tsv_data('/kaggle/input/dakshina-dataset-v1-0/dakshina_dataset_v1.0/hi/lexicons/hi.translit.sampled.test.tsv')

src_vocab, src_inv = build_vocab(train_latin)
tgt_vocab, tgt_inv = build_vocab(train_devnag)

sos_token = tgt_vocab['<sos>']
eos_token = tgt_vocab['<eos>']
pad_idx_src = src_vocab['<pad>']
pad_idx_tgt = tgt_vocab['<pad>']
source_vocab_size = len(src_vocab)
target_vocab_size = len(tgt_vocab)
batch_size = 128
learning_rate = 10**-3
attn_mechanism = True

# Datasets and Loaders
train_dataset = DakshinaCharDataset(train_latin, train_devnag, src_vocab, tgt_vocab)
test_dataset = DakshinaCharDataset(test_latin, test_devnag, src_vocab, tgt_vocab)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                          collate_fn=lambda b: collate_fn(b, pad_idx_src, pad_idx_tgt))
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                        collate_fn=lambda b: collate_fn(b, pad_idx_src, pad_idx_tgt))

# Model, loss, optimizer
encoder = Encoder(config,source_vocab_size).to(device)
decoder = Decoder(config, target_vocab_size,attn_mechanism=attn_mechanism).to(device)
model = Seq2Seq(encoder, decoder, config, target_vocab_size,attn_mechanism=attn_mechanism).to(device)

criterion = nn.CrossEntropyLoss(ignore_index=pad_idx_tgt)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


for epoch in range(config['epochs']):

    #  Training Phase
    model.train()
    total_loss = 0
    
    for src, tgt in train_loader:
        src, tgt = src.to(device), tgt.to(device)
        optimizer.zero_grad()
        output = model(src, tgt)  # [B, T, V]

        output = output[:, 1:].reshape(-1, output.shape[-1])
        tgt_flat = tgt[:, 1:].reshape(-1)
        loss = criterion(output, tgt_flat)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
        
    train_loss = total_loss / len(train_loader)

    print(f"Epoch {epoch+1}/{config['epochs']} | Train Loss: {train_loss:.4f}")

# Testing Phase
model.eval()
test_total_loss = 0
predictions, references = [], []
eos_idx, sos_idx = eos_token, sos_token

with torch.no_grad():
    
    for src, tgt in test_loader:
        src, tgt = src.to(device), tgt.to(device)
        
        # Compute loss
        output = model(src, tgt)
        test_output = output[:, 1:].reshape(-1, output.shape[-1])
        test_tgt = tgt[:, 1:].reshape(-1)
        test_loss = criterion(test_output, test_tgt)
        test_total_loss += test_loss.item()

        # Greedy decode
        batch_size = src.size(0)
        encoder_outputs, hidden = model.encoder(src)
        input_token = torch.full((batch_size,), sos_token, dtype=torch.long, device=src.device)  # shape: [B]
        outputs = []
        attentions = []

        for _ in range(config['max_len']):
            if attn_mechanism:
                output_step, hidden, attn_weights = model.decoder(input_token, hidden, encoder_outputs)
            else:
                output_step, hidden = model.decoder(input_token, hidden, None)

            next_token = output_step.argmax(-1)  # shape: [B]
            outputs.append(next_token.unsqueeze(1))
            attentions.append(attn_weights)  # [B, src_len]
            input_token = next_token
    
        outputs = torch.cat(outputs, dim=1)  # [batch_size, max_len]
        attentions = torch.stack(attentions, dim=1)    # [B, T, src_len]

        decoded =  outputs.tolist()

        for pred, ref in zip(decoded, tgt):
            pred = [i for i in pred if i not in (eos_idx, sos_idx)]
            ref = [i.item() for i in ref if i.item() not in (eos_idx, pad_idx_tgt, sos_idx)]
            predictions.append(pred)
            references.append(ref)

    test_loss_avg = test_total_loss / len(test_loader)
    test_acc = exact_match_accuracy(predictions, references)

    print(f"Test Loss: {test_loss_avg:.4f} | Test Acc: {test_acc:.4f}")

# Save test accuracy in a txt file
with open("test_accuracy_attention.txt", "w") as f:
    f.write(f"Test Loss: {test_loss_avg:.4f}\n")
    f.write(f"Test Accuracy (Exact Match): {test_acc:.4f}\n")

# Save all predictions to 'predictions_vanilla'
output_folder = "predictions_attention"
os.makedirs(output_folder, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_path = os.path.join(output_folder, f"predictions_attention.txt")

with open(output_path, "w", encoding="utf-8") as f:
    f.write("Index\tPrediction\tReference\n")
    f.write("="*40 + "\n")
    for i, (pred, ref) in enumerate(zip(predictions, references)):
        pred_str = ''.join([tgt_inv[idx] for idx in pred])
        ref_str = ''.join([tgt_inv[idx] for idx in ref])
        f.write(f"{i+1}\t{pred_str}\t{ref_str}\n")

# Log creative table to wandb
wandb.init(project="DA6401_Assignment_3",entity="ch21b033-iit-madras", name="Attention_Test_Accuracy")

samples_to_log = 10
table = wandb.Table(columns=["Index", "Prediction", "Reference", "Status"])

for i in range(samples_to_log):
    pred_str = ''.join([tgt_inv[idx] for idx in predictions[i]])
    ref_str = ''.join([tgt_inv[idx] for idx in references[i]])
    status = "✅ Match" if pred_str == ref_str else "❌ Mismatch"
    
    # Add fun Unicode borders or formatting
    pred_decorated = f"{pred_str}"
    ref_decorated = f"{ref_str}"

    table.add_data(i+1, pred_decorated, ref_decorated, status)

# Log the table to wandb
wandb.log({"Sample Attention Predictions Grid": table})
wandb.finish()

# %% Attention heatmap visualization (Question 5)

# Init wandb
wandb.init(project="DA6401_Assignment_3", entity="ch21b033-iit-madras", name="Attention_Heatmap")

# Fetch 10 samples
sample_srcs, sample_tgts = next(iter(test_loader))
sample_srcs, sample_tgts = sample_srcs[:10].to(device), sample_tgts[:10].to(device)

model.eval()
with torch.no_grad():
    encoder_outputs, hidden = model.encoder(sample_srcs)
    input_token = torch.full((10,), sos_token, dtype=torch.long, device=device)

    outputs, attentions = [], []
    for _ in range(config['max_len']):
        output_step, hidden, attn_weights = model.decoder(input_token, hidden, encoder_outputs)
        next_token = output_step.argmax(dim=-1)
        outputs.append(next_token.unsqueeze(1))
        attentions.append(attn_weights)
        input_token = next_token

    outputs = torch.cat(outputs, dim=1)
    attentions = torch.stack(attentions, dim=1)

# 3x3 grid for first 9 samples
fig_grid = make_subplots(
    rows=3,
    cols=3,
    subplot_titles=[f"{i+1}: {''.join([src_inv[t] for t in sample_srcs[i].cpu().tolist() if t != pad_idx_src])} ➡️ {''.join([tgt_inv[t] for t in outputs[i].cpu().tolist() if t not in (pad_idx_tgt, sos_token, eos_token)])}" for i in range(9)],
    horizontal_spacing=0.05,
    vertical_spacing=0.1
)

for idx in range(9):
    src_seq = sample_srcs[idx].cpu().tolist()
    pred_seq = outputs[idx].cpu().tolist()
    attn_matrix = attentions[idx, :len(pred_seq), :len(src_seq)].cpu().numpy()

    src_tokens = [src_inv[i] for i in src_seq if i != pad_idx_src]
    pred_tokens = [tgt_inv[i] for i in pred_seq if i not in (pad_idx_tgt, sos_token, eos_token)]

    row, col = idx // 3 + 1, idx % 3 + 1

    fig_grid.add_trace(
        go.Heatmap(
            z=attn_matrix,
            x=src_tokens,
            y=pred_tokens,
            colorscale='YlOrRd',
            showscale=(idx == 8)
        ),
        row=row, col=col
    )

# Layout adjustments
fig_grid.update_layout(
    height=1000,
    width=1000,
    title_text="Attention Heatmaps Grid (3×3)",
    font=dict(family="Noto Sans Devanagari, Arial Unicode MS, sans-serif", size=10),
    margin=dict(l=30, r=30, t=50, b=30),
    showlegend=False
)

# Save and log HTML
grid_html = "attention_grid_3x3.html"
fig_grid.write_html(grid_html)
wandb.log({"Attention Grid (3x3)": wandb.Html(grid_html)})

# 10th sample plotted separately
idx = 9
src_seq = sample_srcs[idx].cpu().tolist()
pred_seq = outputs[idx].cpu().tolist()
attn_matrix = attentions[idx, :len(pred_seq), :len(src_seq)].cpu().numpy()

src_tokens = [src_inv[i] for i in src_seq if i != pad_idx_src]
pred_tokens = [tgt_inv[i] for i in pred_seq if i not in (pad_idx_tgt, sos_token, eos_token)]

fig10 = go.Figure(data=go.Heatmap(
    z=attn_matrix,
    x=src_tokens,
    y=pred_tokens,
    colorscale='Viridis'
))

fig10.update_layout(
    title=f"Sample 10: {''.join(src_tokens)} ➡️ {''.join(pred_tokens)}",
    xaxis_title="Input (Latin)",
    yaxis_title="Output (देवनागरी)",
    font=dict(family="Noto Sans Devanagari, Arial Unicode MS, sans-serif", size=14),
    width=500,
    height=500
)

fig10.show()
html_path_10 = "attention_sample_10.html"
fig10.write_html(html_path_10)
wandb.log({"Attention Sample 10": wandb.Html(html_path_10)})

wandb.finish()


# %% Connectivity Visualization (Question 6)
def attention_highlight_line(src_tokens, attn_weights, pred_char):
    """
    Highlight source tokens by attention for a single output character.
    """
    cmap = matplotlib.colormaps['Greens']
    html = ""
    for token, weight in zip(src_tokens, attn_weights):
        color = to_hex(cmap(weight))  # convert 0-1 to hex color
        html += f'<span style="background-color:{color}; padding:1px 3px; margin:1px; border-radius:4px;">{token}</span>'
    return f"<div style='font-family:monospace; font-size:18px;'>{html} &nbsp; → <b>{pred_char}</b></div>"

def render_connectivity_visualization(sample_srcs, outputs, attentions, src_inv, tgt_inv, pad_idx_src, pad_idx_tgt, sos_token, eos_token, count=5):
    html_blocks = []

    for idx in range(count):
        src_seq = sample_srcs[idx].cpu().tolist()
        pred_seq = outputs[idx].cpu().tolist()
        attn_matrix = attentions[idx, :len(pred_seq), :len(src_seq)].cpu().numpy()

        src_tokens = [src_inv[i] for i in src_seq if i != pad_idx_src]
        pred_tokens = [tgt_inv[i] for i in pred_seq if i not in (pad_idx_tgt, sos_token, eos_token)]

        rows = []
        for i, pred_char in enumerate(pred_tokens):
            attn_weights = attn_matrix[i, :len(src_tokens)]
            row_html = attention_highlight_line(src_tokens, attn_weights, pred_char)
            rows.append(row_html)

        sentence_html = "<div style='background-color:#1e3a5f; color:white; padding:10px; border-radius:8px; margin-bottom:20px;'>"
        sentence_html += "<br>".join(rows)
        sentence_html += "</div>"

        html_blocks.append(sentence_html)

    full_html = "<html><head><meta charset='UTF-8'></head><body>" + "\n".join(html_blocks) + "</body></html>"
    return full_html


html = render_connectivity_visualization(
    sample_srcs, outputs, attentions,
    src_inv, tgt_inv,
    pad_idx_src, pad_idx_tgt, sos_token, eos_token,
    count=5
)

# Save locally or log to wandb
with open("connectivity_visualization.html", "w", encoding="utf-8") as f:
    f.write(html)

wandb.init(project="DA6401_Assignment_3", entity="ch21b033-iit-madras", name="Attention_Connectivity")

wandb.log({"Connectivity Visualization": wandb.Html("connectivity_visualization.html")})

