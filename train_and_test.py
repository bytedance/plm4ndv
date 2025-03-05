# Copyright (c) 2024 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: MIT
import os
import pickle
import argparse
import numpy as np
import torch
import time
import math
import torch.nn as nn
import torch.nn.functional as F

from extract_parquet import makedirs
from torch.utils.data import Dataset, DataLoader
current_path = os.path.dirname(os.path.abspath(__file__))


makedirs('ckpt')

parser = argparse.ArgumentParser()
parser.add_argument('--layer', type=int, default=1, help='attention')
parser.add_argument('--head', type=int, default=8, help='attention')
parser.add_argument('--sample', type=int, default=1, help='access data or not')
parser.add_argument('--data_path', type=str, default='embedding')
parser.add_argument('--profile_len', type=int, default=100, help='num of rows accessed')

args = parser.parse_args()

BATCH_SIZE = 256
LR = 0.001
HEADS = args.head
PROFILE_SIZE = args.profile_len
emb_path = os.path.join(current_path, f'data/{args.data_path}')
EMB_SIZE = 768

def read_pickle(name):
    with open(os.path.join(emb_path, f'{name}.pkl'), 'rb') as f:
        data = pickle.load(f)
    emb = data['embeddings']
    n = data['N']
    ndv = data['D']
    profile = data['profile']
    tab_cols_cnt = data['tab_cols_cnt']
    return emb, ndv, n, profile, tab_cols_cnt

class NDVDataset(Dataset):
    def __init__(self, embedding, D, N, profile, mask):
        self.embedding = embedding
        self.D = D
        self.N = N
        self.profile = profile
        self.mask = mask

    def __getitem__(self, index):
        return torch.tensor(self.embedding[index], dtype = torch.float32), \
            torch.tensor(self.D[index], dtype = torch.float32), \
                torch.tensor(self.N[index], dtype = torch.float32), \
                    torch.tensor(self.profile[index], dtype = torch.float32), \
                        torch.tensor(self.mask[index], dtype = torch.float32)
    
    def __len__(self):
        return len(self.embedding)

def compute_error(estimated: int, ground_truth: int) -> float:
    if math.isinf(estimated) or estimated == 0:
        err = 1e10
        return err
    assert estimated > 0 and ground_truth > 0, f"estimated and ground_truth NDV must be positive. {estimated}, {ground_truth}"
    err =  max(estimated, ground_truth) / min(estimated, ground_truth)
    if math.isinf(err):
        err = 1e10
    return err

def build_mask_loader(embedding, D, N, profile, tab_cols_cnt, max_column_num, shuffle=False):
    start = 0
    embedding_pad = []
    D_pad = []
    N_pad = []
    profile_pad = []
    mask_list = []
    for column_num in tab_cols_cnt:
        end = start + column_num
        pad_len = max_column_num - column_num
        embedding_pad.append(np.concatenate([embedding[start:end], np.zeros([pad_len, EMB_SIZE])], axis=0))
        D_pad.append(D[start:end] + [1] * pad_len)
        N_pad.append(N[start:end] + [1] * pad_len)
        profile_pad.append(profile[start:end] + np.zeros([pad_len, PROFILE_SIZE + 1]).tolist())
        mask_list.append([1] * column_num + [0] * pad_len)
        start = end
    dataset = NDVDataset(embedding_pad, D_pad, N_pad, profile_pad, mask_list)
    loader = DataLoader(dataset, batch_size = BATCH_SIZE, shuffle = shuffle)
    return loader

def get_dataloaders():
    train_embeddings, train_ndv, train_n, train_profile, train_tab_cnt = read_pickle('train')
    test_embeddings, test_ndv, test_n, test_profile, test_tab_cnt = read_pickle('test')
    val_embeddings, val_ndv, val_n, val_profile, val_tab_cnt = read_pickle('val')
    print(f'embedding loaded')
    max_column_len = max(train_tab_cnt)
    max_column_len = max(max_column_len, max(test_tab_cnt))
    max_column_len = max(max_column_len, max(val_tab_cnt))
    print(f'max column len: {max_column_len}')

    train_loader = build_mask_loader(train_embeddings, train_ndv, train_n, train_profile, train_tab_cnt, max_column_len, shuffle=True)
    print(f'train loader constructed')
    test_loader = build_mask_loader(test_embeddings, test_ndv, test_n, test_profile, test_tab_cnt, max_column_len)
    print(f'test loader constructed')
    val_loader = build_mask_loader(val_embeddings, val_ndv, val_n, val_profile, val_tab_cnt, max_column_len)
    print(f'val loader constructed')

    return train_loader, test_loader, val_loader


def evaluate(model: nn.Module, loader: DataLoader, device, dataset_name:str):
    model.eval()
    predicted_q_error = []

    for embedding, D, N, profile, mask in loader:
        embedding = embedding.to(device)
        N = N.to(device)
        model = model.to(device)
        profile = profile.to(device)
        mask = mask.to(device)

        d = model.inference(embedding, N.unsqueeze(-1), profile, mask).cpu().detach().numpy().tolist()
        D_list = D.cpu().detach().numpy().tolist()
        mask = mask.cpu().detach().numpy().tolist()
        flattern_d = []
        flattern_D = []
        for i, ma in enumerate(mask):
            for j in range(len(ma)):
                if ma[j] == 1:
                    flattern_d.append(d[i][j])
                    flattern_D.append(D_list[i][j])
                if ma[j] == 0:
                    break

        estimated_q_error = [compute_error(flattern_d[i], flattern_D[i]) for i in range(len(flattern_D))]
        predicted_q_error.extend(estimated_q_error)
    print(dataset_name, ':')
    print(f'mean: {np.mean(predicted_q_error):.2f}, 50%: {np.percentile(predicted_q_error, 50):.2f} 75%: {np.percentile(predicted_q_error, 75):.2f} 90%: {np.percentile(predicted_q_error, 90):.2f} 95%: {np.percentile(predicted_q_error, 95):.2f} 99%: {np.percentile(predicted_q_error, 99):.2f}')
    return predicted_q_error

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads=HEADS):
        super(MultiHeadSelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert (
            self.head_dim * num_heads == embed_dim
        ), "Embedding dimension must be divisible by number of heads"

        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.fc_out = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(0.5) 

    def forward(self, embedding_pad, mask):
        batch_size, seq_len, _ = embedding_pad.size()
        Q = self.query(embedding_pad) 
        K = self.key(embedding_pad)     
        V = self.value(embedding_pad)  

        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2) 
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # 计算注意力得分
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5) 
        mask = mask.unsqueeze(1).unsqueeze(2) 
        mask = mask.expand(-1, self.num_heads, -1, -1)  
        mask = mask.expand(-1, -1, seq_len, -1)  

        if mask is not None:
            attn_scores.masked_fill_(mask == 0, float('-inf'))

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights) 

        output = torch.matmul(attn_weights, V) 

        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)

        output = self.fc_out(output) 
        output = self.dropout(output)  
        
        return output, attn_weights


class Model(nn.Module):
    def __init__(self, input_len=768, profile_len=100):
        super(Model, self).__init__()
        # table representation
        self.attentions = nn.ModuleList()
        for i in range(args.layer):
            self.attentions.append(MultiHeadSelfAttention(input_len))
        MultiHeadSelfAttention(input_len)
        if args.sample:
            estimate_input_len = input_len + 1 + profile_len
        else:
            estimate_input_len = input_len + 1
        # NDV estimation
        self.sentence_transform = nn.Sequential(
            nn.Linear(estimate_input_len, 384),
            nn.ReLU(),
            nn.Linear(384, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )
    def run(self, emb, N, profile, mask):
        x = emb
        for attention in self.attentions:
            x, attn_weights = attention(x, mask) 
        emb_transformed = x
        if args.sample:
            x = torch.concat((emb + emb_transformed, torch.log(N), profile[:,:,1:]), dim=-1) # [batch, seq_len, estimate_input_len]
        else:
            x = torch.concat((emb + emb_transformed, torch.log(N)), dim=-1) # [batch, seq_len, without sample]
        ans = self.sentence_transform(x)
        return ans.squeeze()
        
    
    def inference(self, emb, N, profile, mask):
        ans = self.run(emb, N, profile, mask)
        estimate_d = torch.exp(ans)
        return estimate_d.squeeze()
    
    def forward(self, emb, N, profile, mask):
        ans = self.run(emb, N, profile, mask)
        return ans

def train(model, train_loader, test_loader, val_loader, device, model_path):
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.MSELoss(reduction='none')
    epochs = 100
    model = model.to(device)
    print('training...')
    for epoch in range(epochs):
        t0 = time.time()
        model.train()
        for embedding, D, N, profile, mask in train_loader:
            embedding = embedding.to(device)
            N = N.to(device)
            D = D.to(device)
            model = model.to(device)
            profile = profile.to(device)
            mask = mask.to(device)
            optimizer.zero_grad()
            estimate_logd = model(embedding, N.unsqueeze(-1), profile, mask)
            loss = criterion(estimate_logd, torch.log(D))
            masked_loss = (loss * mask).sum(1)
            final_loss = masked_loss.sum() / mask.sum()
            final_loss.backward()
            optimizer.step()
        t1 = time.time()
        
        print(f'training time: {t1-t0:.3f}')
        if epoch % 1 == 0:
            print(f'epoch: [{epoch+1}/{epochs}]')
            predicted_q_error_test = evaluate(model, test_loader, device, 'test')
            predicted_q_error_validation = evaluate(model, val_loader, device, 'validation')
            
            if epoch == 0:
                best_metric = np.percentile(predicted_q_error_validation, 90)
            save_metric = np.percentile(predicted_q_error_validation, 90)
            if save_metric <= best_metric:
                best_metric = save_metric
                print(f'model saved')
                save_epoch = epoch
                torch.save(model.cpu().state_dict(), model_path)
        t2 = time.time()
        print(f'evaluating time: {t2-t1:.3f}', flush=True)
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f'final saved model perf, saved at {save_epoch} epoch')
    evaluate(model, train_loader, device, 'train')
    evaluate(model, val_loader, device, 'validation')
    evaluate(model, test_loader, device, 'test')

def inference(model, test_loader, device, model_path):
    model.load_state_dict(torch.load(model_path, map_location=device))
    evaluate(model, test_loader, device, 'test')


if __name__ == '__main__':
    train_loader, test_loader, val_loader = get_dataloaders()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Model(EMB_SIZE, PROFILE_SIZE)
    if args.sample:
        model_path = os.path.join(current_path, f'ckpt/plm4ndv.pth')
    else:
        model_path = os.path.join(current_path, f'ckpt/plm4ndv_wo_sample.pth')
    print(model)

    # train the model and the parameters will be saved in model_path
    train(model, train_loader, test_loader, val_loader, device, model_path)
    
    # load the model parameter from file to inference
    inference(model, test_loader, device, model_path)