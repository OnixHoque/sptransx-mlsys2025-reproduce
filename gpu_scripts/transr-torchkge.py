# !pip install torchkge py-spy torchviz pykeen

import torch
import pandas as pd
from torch.optim import Adam
torch.manual_seed(42)
from torchkge.models import *
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

model_name = 'transr'
framework = 'torchkge'
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
emb_dim = 128
rel_dim = 128
batch_fact = 2
# rel_dim = 32
lr = 0.0004
n_epochs = 200
b_size = 32768 * batch_fact
margin = 0.5

# Define the model and criterion
model = TransRModel(emb_dim, rel_dim, kg_train.n_ent, kg_train.n_rel)
criterion = MarginLoss(margin)

# Move everything to CUDA if available
if cuda.is_available():
    cuda.empty_cache()
    model.cuda()
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


torch.autograd.set_detect_anomaly(True)
loss_kge = []
total_forward_time = 0
total_backward_time = 0

total_training_time = time.time()

for m in tqdm(range(n_epochs)):
    fw_start = time.time()
    optimizer.zero_grad()
    pos, neg = model(h_idx, t_idx, r_idx, n_h_idx, n_t_idx)
    loss = criterion(pos, neg)
    total_forward_time += (time.time() - fw_start)
    
    bw_start = time.time()
    loss.backward()
    loss_kge += [loss.item()]
    total_backward_time += (time.time() - bw_start)
    
    optimizer.step()
    
total_training_time = time.time() - total_training_time
    
final_peak_memory = torch.cuda.max_memory_allocated('cuda:0') / 1e9

output_str = f'{framework}\t{model_name}\t{dataset_name}\t{num_batches}\t{total_training_time}\t{total_forward_time}\t{total_backward_time}\t{final_peak_memory}\n'

print(output_str)
with open(f'output/{framework}-{model_name}-{dataset_name}.txt', 'w') as f:
    f.write(output_str)
    
    