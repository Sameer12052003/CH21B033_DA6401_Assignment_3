import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import wandb
import numpy as np
from data_preprocessing.data_preprocess import exact_match_accuracy, greedy_decode
from data_preprocessing.data_preprocess import (
    build_vocab, DakshinaCharDataset, collate_fn, load_tsv_data
)
from seq2seqmodel.model import Encoder, Decoder, Seq2Seq

np.random.seed(42)

# WandB Sweep Config
sweep_config = {
    'method': 'bayes',
    'name': 'DL-Att-Assn-3-hyperparam-search-1',
    'metric': {'name': 'val_accuracy', 'goal': 'maximize'},
    'parameters': {
        'embedding_dim': {'values': [64,128,256]},
        'hidden_size': {'values': [64, 128, 256]},
        'num_layers': {'values': [2,3]},
        'dropout': {'values': [0.25, 0.3]},
        'cell_type' : {'values' : ['RNN','GRU', 'LSTM']},
        'beam_search' : {'values': [1]},
        'epochs': {'value': 15},
        'max_len': {'value': 30}
    }
}


def train():
    
    wandb.init(project="DA6401_Assignment_3",entity="ch21b033-iit-madras",
               name= "Hyperparam_Search_1")
    
    config = wandb.config
    
     # Re-assign just to ensure it reflects in plots/legends
    wandb.run.name = f"""DL_sweep_2_cell_type_{config.cell_type}_emd_dim_{config.embedding_dim}_hs_{config.hidden_size}
            num_layers_{config.num_layers}_bs_{config.beam_search}"""
            
    wandb.run.save()  # ensures name update is logged
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load and tokenize data
    
    train_latin, train_devnag = load_tsv_data('/kaggle/input/dakshina-dataset-v1-0/dakshina_dataset_v1.0/hi/lexicons/hi.translit.sampled.train.tsv')
    val_latin, val_devnag = load_tsv_data('/kaggle/input/dakshina-dataset-v1-0/dakshina_dataset_v1.0/hi/lexicons/hi.translit.sampled.dev.tsv')

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
    val_dataset = DakshinaCharDataset(val_latin, val_devnag, src_vocab, tgt_vocab)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              collate_fn=lambda b: collate_fn(b, pad_idx_src, pad_idx_tgt))
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            collate_fn=lambda b: collate_fn(b, pad_idx_src, pad_idx_tgt))

    # Model, loss, optimizer
    encoder = Encoder(config,source_vocab_size).to(device)
    decoder = Decoder(config, target_vocab_size,attn_mechanism=attn_mechanism).to(device)
    model = Seq2Seq(encoder, decoder, config, target_vocab_size).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx_tgt)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


    for epoch in range(config.epochs):
    
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

        # 🔍 Validation Phase
        model.eval()
        val_total_loss = 0
        predictions, references = [], []
        eos_idx, sos_idx = eos_token, sos_token

        with torch.no_grad():
            
            for src, tgt in val_loader:
                src, tgt = src.to(device), tgt.to(device)
                
                # Compute loss
                output = model(src, tgt)
                val_output = output[:, 1:].reshape(-1, output.shape[-1])
                val_tgt = tgt[:, 1:].reshape(-1)
                val_loss = criterion(val_output, val_tgt)
                val_total_loss += val_loss.item()

                # Greedy decode
                batch_size = src.size(0)
                encoder_outputs, hidden = model.encoder(src)
                input_token = torch.full((batch_size,), sos_token, dtype=torch.long, device=src.device)  # shape: [B]
                outputs = []
        
                for _ in range(config.max_len):
                    if attn_mechanism:
                        output_step, hidden, attn_wts = model.decoder(input_token, hidden, encoder_outputs)
                    else:
                        output_step, hidden = model.decoder(input_token, hidden, None)

                    next_token = output_step.argmax(-1)  # shape: [B]
                    outputs.append(next_token.unsqueeze(1))
                    input_token = next_token
            
                outputs = torch.cat(outputs, dim=1)  # [batch_size, max_len]
                decoded =  outputs.tolist()

                for pred, ref in zip(decoded, tgt):
                    pred = [i for i in pred if i not in (eos_idx, sos_idx)]
                    ref = [i.item() for i in ref if i.item() not in (eos_idx, pad_idx_tgt, sos_idx)]
                    predictions.append(pred)
                    references.append(ref)

        val_loss_avg = val_total_loss / len(val_loader)
        val_acc = exact_match_accuracy(predictions, references)

        print(f"Epoch {epoch+1}/{config.epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss_avg:.4f} | Val Acc: {val_acc:.4f}")
        wandb.log({
            "train_loss": train_loss,
            "val_loss": val_loss_avg,
            "val_accuracy": val_acc
        })


sweep_id = wandb.sweep(sweep_config, project="DA6401_Assignment_3",entity="ch21b033-iit-madras")
wandb.agent(sweep_id, function=train,count=20)
