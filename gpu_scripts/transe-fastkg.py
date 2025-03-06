# !pip install torchkge py-spy torchviz pykeen

import torch
import pandas as pd
from torch.optim import Adam
torch.manual_seed(42)
from torchkge.models import TransEModel, TorusEModel
from torchkge.sampling import BernoulliNegativeSampler
from torchkge.utils import MarginLoss, DataLoader
import pickle
import gc
from tqdm.autonotebook import tqdm
from torch import cuda
from torchkge.utils.datasets import *
from torch.nn.functional import normalize
import torch_sparse
import dgl.sparse as dglsp
from torch_sparse import SparseTensor
import time, sys

torch.cuda.empty_cache()

model_name = 'transe'
framework = 'fastkg'
dataset_name = sys.argv[1]


def load_biokg_dataset():
    with open('./dataset/biokg-torchkge.pkl', 'rb') as f:
        ret = pickle.load(f)
    return ret, None, None

from torchkge.utils.datasets import *
dataset_loader_map = {
    'fb15k': load_fb15k,
    'fb15k237': load_fb15k237,
    'biokg': load_biokg_dataset,
    'fb13': load_fb13,
    'wn18': load_wn18,
    'wn18rr': load_wn18rr,
    'yago3': load_yago3_10,
}

kg_train, _, _ = dataset_loader_map[dataset_name]()

# Define some hyper-parameters for training
emb_dim = 1024
rel_dim = 128
batch_fact = 12
# rel_dim = 32
lr = 0.0004
n_epochs = 200
b_size = 32768 * batch_fact
margin = 0.5

# Define the model and criterion
model = TransEModel(emb_dim, kg_train.n_ent, kg_train.n_rel, dissimilarity_type='L2')
criterion = MarginLoss(margin)

# Move everything to CUDA if available
if cuda.is_available():
    cuda.empty_cache()
    # model.cuda()
    criterion.cuda()

# Define the torch optimizer to be used
optimizer = Adam(model.parameters(), lr=lr, weight_decay=1e-5)

sampler = BernoulliNegativeSampler(kg_train)
dataloader = DataLoader(kg_train, batch_size=b_size, use_cuda='all')
num_batches = len(dataloader)

data_contents = [(i, batch) for i, batch in enumerate(dataloader)]
h_idx, t_idx, r_idx = data_contents[0][1]
h_idx, t_idx, r_idx = h_idx.clone(), t_idx.clone(), r_idx.clone()

n_h_idx, n_t_idx = sampler.corrupt_batch(h_idx, t_idx, r_idx)

h_idx.device

# ent_emb = model.ent_emb.weight.data.clone().detach().to('cuda:0').requires_grad_(True)
# rel_emb = model.rel_emb.weight.data.clone().detach().to('cuda:0').requires_grad_(True)

ent_emb = model.ent_emb.weight.data.clone().detach().to('cuda:0')
rel_emb = model.rel_emb.weight.data.clone().detach().to('cuda:0')


del model
gc.collect()

ent_emb.data = normalize(ent_emb.data, p=2, dim=1)
rel_emb.data = normalize(rel_emb.data, p=2, dim=1)

all_emb = torch.cat((ent_emb.data, rel_emb.data), dim=0).requires_grad_()
self_optimizer = torch.optim.Adam([all_emb], lr=lr, weight_decay=1e-5)

ENGINE='dgl'

def generate_sparse_matrix(indices, values, size, engine=ENGINE):
    if engine is None:
        return torch.sparse_coo_tensor(
                    indices=indices,
                    values=values,
                    size=size
                )._requires_grad(False)
    elif engine == 'torchsparse':
        return SparseTensor(
                row=indices[0],
                col=indices[1],
                value=values,
                sparse_sizes=size
            ).requires_grad_(False)
    elif engine == 'dgl':
        return dglsp.spmatrix(indices, values.requires_grad_(False), size)

def perform_spmm(adj, dense, engine=ENGINE):
    if engine is None:
        return torch.sparse.mm(adj, dense)
    elif engine == 'torchsparse':
        return torch_sparse.matmul(adj, dense)
    elif engine == 'dgl':
        return dglsp.spmm(adj, dense)
    
bt_size = h_idx.shape[0]
adj_mat_idx = torch.tensor(range(bt_size), device=ent_emb.device).repeat(3)

def get_adj1():
  bt_size = h_idx.shape[0]
  return generate_sparse_matrix(
      indices=torch.stack([
            adj_mat_idx.data,
            torch.cat([h_idx.data, t_idx.data, r_idx + kg_train.n_ent])
        ]),
      values=torch.cat([
            torch.full((len(h_idx),), 1, dtype=ent_emb.dtype, device=ent_emb.device),
            torch.full((len(t_idx),), -1, dtype=ent_emb.dtype, device=ent_emb.device),
            torch.full((len(r_idx),), 1, dtype=ent_emb.dtype, device=ent_emb.device),
        ]),
      size=(bt_size, kg_train.n_ent + kg_train.n_rel)
      )

def get_adj1_neg():
  bt_size = h_idx.shape[0]
  return generate_sparse_matrix(
      indices=torch.stack([
            adj_mat_idx.data,
            torch.cat([n_h_idx.data, n_t_idx.data, r_idx + kg_train.n_ent])
        ]),
      values=torch.cat([
            torch.full((len(n_h_idx),), 1, dtype=ent_emb.dtype, device=ent_emb.device),
            torch.full((len(n_t_idx),), -1, dtype=ent_emb.dtype, device=ent_emb.device),
            torch.full((len(r_idx),), 1, dtype=ent_emb.dtype, device=ent_emb.device),
        ]),
      size=(bt_size, kg_train.n_ent + kg_train.n_rel)
      )


adj1 = get_adj1()
adj1_neg = get_adj1_neg()

def train():
  global adj1, adj2, adj1_neg
  global ent_emb, rel_emb, all_emb
  with torch.no_grad():
    all_emb.data = normalize(all_emb.data, p=2, dim=1)
    
  # Pos
  x = perform_spmm(adj1, all_emb)
  p = -1 * torch.linalg.vector_norm(x, ord=2, dim=-1)**2

  # # Neg
  x_ = perform_spmm(adj1_neg, all_emb)  
  n = -1 * torch.linalg.vector_norm(x_, ord=2, dim=-1)**2

  return p, n

torch.autograd.set_detect_anomaly(True)
loss_lin_alg = []
total_forward_time = 0
total_backward_time = 0

total_training_time = time.time()

for m in tqdm(range(n_epochs)):
    fw_start = time.time()
    self_optimizer.zero_grad()
    a, b = train()
    c = criterion(a, b)
    total_forward_time += (time.time() - fw_start)
    
    bw_start = time.time()
    c.backward()
    loss_lin_alg += [c.item()]
    total_backward_time += (time.time() - bw_start)
    
    self_optimizer.step()
    
total_training_time = time.time() - total_training_time
    
final_peak_memory = torch.cuda.max_memory_allocated('cuda:0') / 1e9

output_str = f'{framework}\t{model_name}\t{dataset_name}\t{num_batches}\t{total_training_time}\t{total_forward_time}\t{total_backward_time}\t{final_peak_memory}\n'

print(output_str)
with open(f'output/{framework}-{model_name}-{dataset_name}.txt', 'w') as f:
    f.write(output_str)