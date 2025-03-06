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
framework = 'pyg'
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
# criterion = MarginLoss(margin)

# Move everything to CUDA if available
if cuda.is_available():
    cuda.empty_cache()
    # model.cuda()
    # criterion.cuda()

# Define the torch optimizer to be used
# optimizer = Adam(model.parameters(), lr=lr, weight_decay=1e-5)

sampler = BernoulliNegativeSampler(kg_train)
dataloader = DataLoader(kg_train, batch_size=b_size, use_cuda='all')
num_batches = len(dataloader)

data_contents = [(i, batch) for i, batch in enumerate(dataloader)]
h_idx, t_idx, r_idx = data_contents[0][1]
h_idx, t_idx, r_idx = h_idx.clone(), t_idx.clone(), r_idx.clone()

n_h_idx, n_t_idx = sampler.corrupt_batch(h_idx, t_idx, r_idx)

h_idx.device


ent_emb = model.ent_emb.weight.data.clone().detach()
rel_emb = model.rel_emb.weight.data.clone().detach()


from torch_geometric.nn import TransE

device = 'cuda' if torch.cuda.is_available() else 'cpu'
pyg_model = TransE(
    num_nodes=kg_train.n_ent,
    num_relations=kg_train.n_rel,
    hidden_channels=1024,
    margin=0.5,
    p_norm=2
).to(device)

pyg_model.node_emb.weight.data.copy_(ent_emb.data)
pyg_model.rel_emb.weight.data.copy_(rel_emb.data)

del model
del ent_emb
del rel_emb
gc.collect()

pyg_optimizer = Adam(
    pyg_model.parameters(),
    lr=0.0004,
    weight_decay=1e-5
)

pyg_model.train()

# pyg_loss_fn = MarginLoss(margin=0.5)
pyg_criterion = MarginLoss(margin)
pyg_criterion.cuda()



torch.autograd.set_detect_anomaly(True)
pyg_loss = []
total_forward_time = 0
total_backward_time = 0

total_training_time = time.time()

for m in tqdm(range(n_epochs)):
    fw_start = time.time()
    pyg_optimizer.zero_grad()
    pos_score = pyg_model(h_idx, r_idx, t_idx)
    neg_score = pyg_model(n_h_idx, r_idx, n_t_idx)
    loss = pyg_criterion(pos_score, neg_score)
    total_forward_time += (time.time() - fw_start)
    
    bw_start = time.time()
    loss.backward()
    pyg_loss += [loss.item()]
    total_backward_time += (time.time() - bw_start)
    
    pyg_optimizer.step()
    
total_training_time = time.time() - total_training_time
    
final_peak_memory = torch.cuda.max_memory_allocated('cuda:0') / 1e9

output_str = f'{framework}\t{model_name}\t{dataset_name}\t{num_batches}\t{total_training_time}\t{total_forward_time}\t{total_backward_time}\t{final_peak_memory}\n'

print(output_str)
with open(f'output/{framework}-{model_name}-{dataset_name}.txt', 'w') as f:
    f.write(output_str)
    
    