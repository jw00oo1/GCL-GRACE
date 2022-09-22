# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
# %%
def masking_attr(node_feature, p:float):
    assert p >= 0. and p <= 1.
    
    feature_dim = node_feature.size(dim=-1)
    mask = torch.rand(feature_dim, device=node_feature.device) >= p
    
    masked_feature = node_feature.clone()
    masked_feature[:, mask] = 0.
    
    return masked_feature
# %%
class Encoder(nn.Module):
    def __init__(self, in_channels:int, out_channels:int, activation, k:int=2):
        super(Encoder, self).__init__()
        
        conv_list = [GCNConv(in_channels,2*out_channels)]
        for _ in range(k-2):
            conv_list.append(GCNConv(in_channels,2*out_channels))
        conv_list.append(GCNConv(2*out_channels, out_channels))
        self.conv_list = nn.ModuleList(conv_list)
        
        self.ac = activation
        self.k = k
        self.outdim = out_channels
        
    def forward(self, x:torch.Tensor, A:torch.Tensor):
        for i in range(self.k):
            x = self.conv_list[i](x, A)
            x = self.ac(x)
        return x
    
    def get_outdim(self):
        return self.outdim
# %%
# graph -> embedding, loss
class GCL(nn.Module):
    def __init__(self, encoder:Encoder, hidden_dim:int, tau:float, activation):
        super(GCL, self).__init__()
        
        self.encoder = encoder
        self.tau = tau
        
        embedding_dim = encoder.get_outdim()
        self.g = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            activation,
            nn.Linear(hidden_dim, embedding_dim)
        )
    
    def forward(self, x, A):
        return self.encoder(x, A)
    
    def sim(self, z1, z2):
        z1 = F.normalize(self.g(z1))
        z2 = F.normalize(self.g(z2))
        
        return torch.matmul(z1, z2.t())
    
    # int : (num_node, embedding_dim), out : (num_node, )
    def semi_loss(self, z1, z2)->torch.Tensor:
        intra_sim = torch.exp(self.sim(z1, z1) / self.tau)
        inter_sim = torch.exp(self.sim(z1, z2) / self.tau)
        
        return -torch.log(inter_sim.diag() / 
                         (inter_sim.sum(dim=1) + intra_sim.sum(dim=1) - intra_sim.diag()))
        
    def batch_semi_loss(self, z1, z2, batch_size)->torch.Tensor:
        num_nodes = z1.size(0)
        num_batches = (num_nodes-1) // batch_size + 1
        indices = torch.arange(0, num_nodes)
        losses = []
        
        for i in range(num_batches):
            batch_indices = indices[i*batch_size:(i+1)*batch_size]
            batch_z1 = z1[batch_indices]
            
            intra_sim = torch.exp(self.sim(batch_z1, z1) / self.tau)
            inter_sim = torch.exp(self.sim(batch_z1, z2) / self.tau)
            
            losses.append(
                    -torch.log(inter_sim[:, batch_indices]) /
                    (inter_sim.sum(dim=1) + intra_sim.sum(dim=1) - intra_sim[:, batch_indices])
                )
        
        return torch.Tensor(losses).unsqueeze(dim=1)
    
    def loss(self, z1, z2, batch_size=0):       
        if batch_size == 0:
            l1 = self.semi_loss(z1,z2)
            l2 = self.semi_loss(z2,z1)
        else:
            l1 = self.batch_semi_loss(z1,z2,batch_size)
            l2 = self.batch_semi_loss(z2,z1,batch_size)
            
        loss = (l1 + l2) / 2
        loss = loss.mean()
        
        return loss

# %%
