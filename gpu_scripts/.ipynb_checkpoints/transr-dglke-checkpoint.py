# !pip install torchkge py-spy torchviz pykeen
import torch
import pandas as pd
# from torch.optim import Adam
from tqdm.autonotebook import tqdm
from torch import cuda
torch.manual_seed(42)
import time, sys

# from torchkge.models import TransEModel, TorusEModel
# from torchkge.sampling import BernoulliNegativeSampler
# from torchkge.utils import MarginLoss, DataLoader
# import pickle
# import gc

# from torchkge.utils.datasets import *
# from torch.nn.functional import normalize
# import torch_sparse
# import dgl.sparse as dglsp
# from torch_sparse import SparseTensor

torch.cuda.empty_cache()

model_name = 'transr'
framework = 'dglke'
dataset_name = sys.argv[1]

batch_fact = 2
emb_dim = 128
n_epochs = 200
num_batches = 0

from dglke.models import KEModel
class MyArgs:
    pass
arg_object = MyArgs()
arg_object.has_edge_importance = False
arg_object.gpu = [0]
arg_object.mix_cpu_gpu = False
arg_object.strict_rel_part = False
arg_object.soft_rel_part = False
arg_object.dataset = dataset_name
arg_object.neg_deg_sample = False
arg_object.regularization_coef = 0.0
arg_object.lr = 0.0004
arg_object.margin = 0.5
arg_object.loss_genre = 'Margin'


def get_dataset_from_name():
    global dataset_name
    if dataset_name == 'fb15k':
        return get_dataset('./dataset', data_name='FB15k', format_str = 'built_in')
    if dataset_name == 'fb15k237':
        return get_dataset('./dataset', data_name='FB15k-237', format_str = 'built_in')
    if dataset_name == 'biokg':
        return get_dataset('./dataset', data_name='biokg', format_str = 'built_in')
    if dataset_name == 'fb13':
        return get_dataset('/global/homes/m/mdshoque/torchkge_data/FB13', data_name='FB13', files=['train2id.txt', 'valid2id.txt', 'test2id.txt'], format_str = 'raw_udd_hrt')
    if dataset_name == 'wn18':
        return get_dataset('./dataset', data_name='wn18', format_str = 'built_in')
    if dataset_name == 'wn18rr':
        return get_dataset('./dataset', data_name='wn18rr', format_str = 'built_in')
    if dataset_name == 'yago3':
        return get_dataset('/global/homes/m/mdshoque/torchkge_data/YAGO3-10', data_name='YAGO3-10', files=['train.txt', 'valid.txt', 'test.txt'], format_str = 'raw_udd_hrt')

from dglke.dataloader import get_dataset
fb15k = get_dataset_from_name()

dglke_model =KEModel(arg_object, 'TransR', n_entities=fb15k.n_entities, n_relations=fb15k.n_relations, hidden_dim=emb_dim, gamma=0.5, double_entity_emb=False, double_relation_emb=False)
print(fb15k.n_entities, fb15k.n_relations)

from dglke.dataloader import ConstructGraph
g = ConstructGraph(fb15k, arg_object)
g.to('cuda:0')

from dglke.dataloader import TrainDataset, NewBidirectionalOneShotIterator

train_data = TrainDataset(g, fb15k, arg_object, ranks=0, has_importance=arg_object.has_edge_importance)
train_data

neg_sample_size = 1
train_sampler_head = train_data.create_sampler(batch_size=32768 * batch_fact,
                                                       neg_sample_size=neg_sample_size,
                                                       neg_chunk_size=neg_sample_size,
                                                       mode='head',
                                                       num_workers=1,
                                                       shuffle=True,
                                                       exclude_positive=False)

train_sampler_tail = train_data.create_sampler(batch_size=32768 * batch_fact,
                                                       neg_sample_size=neg_sample_size,
                                                       neg_chunk_size=neg_sample_size,
                                                mode='tail',
                                                num_workers=1,
                                                shuffle=True,
                                                exclude_positive=False)
train_sampler = NewBidirectionalOneShotIterator(train_sampler_head, train_sampler_tail,
                                                neg_sample_size, neg_sample_size,
                                                True, fb15k.n_entities,
                                                arg_object.has_edge_importance)

p, n = next(train_sampler)



dgl_loss = []
total_forward_time = 0
total_backward_time = 0

total_training_time = time.time()

for _ in tqdm(range(n_epochs)):
    fw_start = time.time()
    loss, log = dglke_model.forward(p, n, 0)
    total_forward_time += (time.time() - fw_start)
    
    bw_start = time.time()
    loss.backward()
    dgl_loss += [loss.item()]
    total_backward_time += (time.time() - bw_start)
    
    dglke_model.update()
    
total_training_time = time.time() - total_training_time
    
final_peak_memory = torch.cuda.max_memory_allocated('cuda:0') / 1e9

output_str = f'{framework}\t{model_name}\t{dataset_name}\t{num_batches}\t{total_training_time}\t{total_forward_time}\t{total_backward_time}\t{final_peak_memory}\n'

print(output_str)
with open(f'output/{framework}-{model_name}-{dataset_name}.txt', 'w') as f:
    f.write(output_str)