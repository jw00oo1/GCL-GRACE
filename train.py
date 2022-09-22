# %%
import torch
from model import *
from torch_geometric.datasets import Planetoid
from torch_geometric.utils.dropout import dropout_adj
import torch.nn.functional as F
from eval import label_classification
import matplotlib.pyplot as plt

from tqdm import tqdm

# %%
drop_feature_rate_1 = 0.2
drop_feature_rate_2 = 0.4
drop_edge_rate_1 = 0.3
drop_edge_rate_2 = 0.4
tau = 0.4
num_epochs = 200
weight_decay = 1e-5
lr = 5e-4
embedding_dim = 128
hidden_dim = 128
activation = nn.ReLU()

# %%
def train(model:GCL, x, edge_index):
    model.train()
    x1_drop = masking_attr(x, drop_feature_rate_1)
    x2_drop = masking_attr(x, drop_feature_rate_2)
    edge1_drop = dropout_adj(edge_index, p=drop_edge_rate_1)[0]
    edge2_drop = dropout_adj(edge_index, p=drop_edge_rate_2)[0]
    
    z1 = model(x1_drop, edge1_drop)
    z2 = model(x2_drop, edge2_drop)
    
    optimizer.zero_grad()
    loss = model.loss(z1, z2)
    loss.backward()
    optimizer.step()
    
    return loss.item()

# %%
def test(model:GCL, x, edge_index, y):
    model.eval()
    z = model(x, edge_index)
    
    acc_dict = label_classification(z, y, 0.8)
    
    return acc_dict

# %%
device = 'cuda' if torch.cuda.is_available() else 'cpu'

dataset = Planetoid(root='./data/Cora',name='Cora')
data = dataset[0].to(device)

# %%
encoder = Encoder(dataset.num_features, embedding_dim, activation).to(device)
model = GCL(encoder, hidden_dim, tau, activation).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

# %%
# train
gcl_losses = []

pbar = tqdm(range(num_epochs))
for epoch in pbar:
    loss = train(model, data.x, data.edge_index)
    
    gcl_losses.append(loss)
    loss_msg = f'epoch : {epoch:03d}, loss : {loss:.4f}'
    pbar.set_description(loss_msg)

#test
# raw_acc_dict = test()
gcl_acc_dict = test(model, data.x, data.edge_index, data.y)

for key, val in gcl_acc_dict.items():
    print(f'{key} : {val:.04f}')

plt.plot(gcl_losses)
plt.xlabel('epoches')
plt.ylabel('gcl loss')
plt.savefig('gcl_loss.png')
# %%
