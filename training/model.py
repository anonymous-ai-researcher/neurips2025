
import numpy as np
import torch
import torch.nn as nn
import math
    
class DFALC(nn.Module):
    def __init__(self, params, conceptSize, roleSize, cEmb_init, rEmb_init,  device, name="Godel"):
        super().__init__()
        self.params = params
        self.conceptSize, self.roleSize = conceptSize, roleSize
        self.device = device
        self.cEmb = nn.Parameter(torch.tensor(cEmb_init))
        self.rEmb = nn.Parameter(torch.tensor(rEmb_init))
        self.relu = torch.nn.ReLU()
        # self.c_mask, self.r_mask = self.get_mask()
        self.logic_name = name
        self.epsilon = 1e-2
        self.p=2
        self.tau = 0.1  
        self.box_dim = 2  
        self.gamma = nn.Parameter(torch.tensor(0.5))  
        self.value_clamp = 1e-3


    def to_sparse(self, A):
        return torch.sparse_coo_tensor(np.where(A!=0),A[np.where(A!=0)],A.shape)
    
    def index_sparse(self, A, idx):
        return torch.where(A.indices[0] in idx)
    
    def pi_0(self, x):
        return (1-self.epsilon)*x+self.epsilon
    
    def pi_1(self, x):
        return (1-self.epsilon)*x
    
    
    def neg(self, x, negf):
        negf = negf.unsqueeze(1)
        negf2 = negf*(-2) + 1

        
        return negf2*x
        
    def t_norm(self, x, y):
        if self.logic_name == "Godel" or "Rule":
            return torch.minimum(x,y)
        elif self.logic_name == "LTN":
            return self.pi_0(x)*self.pi_0(y)
        # elif self.logic_name == "Product":
        #     return x*y

        elif self.logic_name == "Falcon":
            return torch.maximum(x + y - 1, torch.tensor(0.0, device=x.device))

        elif self.logic_name == "BoxEL":
            epsilon = 1e-3
            return ((1-epsilon)*x+epsilon)*((1-epsilon)*y+epsilon)
        elif self.logic_name == "Box2EL":
            return torch.maximum(x + y - 1, torch.tensor(0.0, device=x.device))

        elif self.logic_name == "ELEmbedding":
            x = torch.clamp(x, self.value_clamp, 1.0 - self.value_clamp)
            y = torch.clamp(y, self.value_clamp, 1.0 - self.value_clamp)
            numerator = x * y
            denominator = self.gamma + (1 - self.gamma) * (x + y - x * y)
            return numerator / (denominator + 1e-8)
    def t_cnorm(self, x, y):
        if self.logic_name == "Godel" or "Rule":
            return torch.maximum(x,y)
        elif self.logic_name == "LTN":
            a = self.pi_1(x)
            b = self.pi_1(y)
            return a+b-a*b
        elif self.logic_name == "Falcon":
            return torch.minimum(x+y,torch.tensor(1.0, device=x.device))
        elif self.logic_name == "BoxEL":
            epsilon = 1e-3
            a = (1-epsilon)*x
            b = (1-epsilon)*y
            return a+b-a*b
        elif self.logic_name == "Box2EL":
            return torch.minimum(x+y,torch.tensor(1.0, device=x.device))
        elif self.logic_name == "ELEmbedding":
            x = torch.clamp(x, self.value_clamp, 1.0 - self.value_clamp)
            y = torch.clamp(y, self.value_clamp, 1.0 - self.value_clamp)
            numerator = x + y - (2 - self.gamma) * x * y
            denominator = 1 - (1 - self.gamma) * x * y
            return numerator / (denominator + 1e-8)

    def forall(self, r, x):
        if self.logic_name == "Godel" or "Rule":
            return torch.min(self.t_cnorm(1-r,x.unsqueeze(1).expand(r.shape)),2).values
        elif self.logic_name == "LTN":
            return 1-torch.pow(torch.mean(torch.pow(1-self.pi_1(self.t_cnorm(r,x.unsqueeze(1).expand(r.shape))),self.p),2),1/self.p)
        elif self.logic_name == "Falcon":
            p_values = self.t_cnorm(r, x.unsqueeze(1).expand(r.shape))
            return torch.min(p_values, dim=1).values 
        elif self.logic_name == "BoxEL":
            # print("here: ",r,x) 
            values = torch.pow(torch.mean(torch.pow(1-self.pi_1(self.t_cnorm(r,x.unsqueeze(1).expand(r.shape))),self.p),2),1/self.p)
            return 1-values
        elif  self.logic_name == "Box2EL":
            p_values = self.t_cnorm(r, x.unsqueeze(1).expand(r.shape))
            return torch.min(p_values, dim=1).values
        elif self.logic_name == "ELEmbedding":
            expanded_x = x.unsqueeze(1).expand(r.shape)
            cnorm_values = self.t_cnorm(1 - r, expanded_x)
            return torch.min(cnorm_values, dim=1).values
    
    def exist(self, r, x):
        if self.logic_name == "Godel" or "Rule":
            return torch.max(self.t_norm(r,x.unsqueeze(1).expand(r.shape)),2).values
        elif self.logic_name == "LTN":
            return torch.pow(torch.mean(torch.pow(self.pi_0(self.t_norm(r,x.unsqueeze(1).expand(r.shape))),self.p),2),1/self.p)
        elif self.logic_name == "Falcon":
            p_values = self.t_norm(r, x.unsqueeze(1).expand(r.shape))
            return torch.max(p_values, dim=1).values  
        elif self.logic_name == "BoxEL":
            return torch.pow(torch.mean(torch.pow(self.pi_0(self.t_norm(r,x.unsqueeze(1).expand(r.shape))),self.p),2),1/self.p)
        elif self.logic_name == "Box2EL":
            p_values = self.t_norm(r, x.unsqueeze(1).expand(r.shape))
            return torch.max(p_values, dim=1).values 
        elif self.logic_name == "ELEmbedding":
            expanded_x = x.unsqueeze(1).expand(r.shape)
            tnorm_values = self.t_norm(r, expanded_x)
            return torch.max(tnorm_values, dim=1).values 
    def L2(self, x, dim=1):
        return torch.sqrt(torch.sum((x)**2, dim))
    
    def L2_dist(self, x, y, dim=1):
        return torch.sqrt(torch.sum((x-y)**2, dim))
    
    def L1(self,x,dim=1):
        return torch.sum(torch.abs(x),dim)
    
    def L1_dist(self,x,y,dim=1):
        return torch.sum(torch.abs(x-y),dim)
    
    def HierarchyLoss(self, lefte, righte):
        return torch.mean(self.L1(self.relu(lefte-righte)))

    def rule_based_loss(self, lefte, atype, righte, left, right, negf):
        loss = 0
        
        if atype == 0:                
            loss = torch.mean(self.L1((1-righte)*self.relu(lefte-righte)))
                
        elif atype == 1:              
            loss = torch.mean(self.L1((1-righte)*self.relu(lefte-righte)))

        elif atype == 2:
            loss = torch.mean(self.L1((1-righte)*self.relu(lefte-righte)))

        elif atype == 3:
            loss = torch.mean(self.L1((1-lefte)*self.relu(0.8-lefte)*self.relu(torch.sum(torch.minimum(righte.unsqueeze(-1),self.rEmb[right[:,0]]),dim=1)-0.8)))

        elif atype == 4:                
            loss = torch.mean(self.L1((1-lefte)*self.relu(0.8-lefte)*self.relu(torch.sum(torch.minimum(righte.unsqueeze(-1),self.rEmb[right[:,0]]),dim=1)-0.8)))
                
        elif atype == 5:              
            loss = torch.mean(self.L1((1-righte)*self.relu(0.8-righte)*self.relu(torch.sum(torch.minimum(lefte.unsqueeze(-1),self.rEmb[left[:,0]]),dim=2)-0.8)))

        elif atype == 6:
            loss = torch.mean(self.L1((1-righte)*self.relu(0.8-righte)*self.relu(torch.sum(torch.minimum(lefte.unsqueeze(-1),self.rEmb[left[:,0]]),dim=2)-0.8) + (1-lefte)*self.relu(0.8-lefte)*self.relu(torch.sum(torch.minimum(righte.unsqueeze(-1),self.rEmb[left[:,0]]),dim=1)-0.8)))


        return loss
        
        
        

    def forward(self, batch, atype, device):
        left, right, negf = batch
        
        loss, lefte, righte, b_c_mask, b_r_mask = None, None, None, None, None
        
        self.cEmb[-1,:].detach().masked_fill_(self.cEmb[-1,:].gt(0.0),1.0)
        self.cEmb[-2,:].detach().masked_fill_(self.cEmb[-2,:].lt(1),0.0)
        
        
        if atype == 0:
            lefte = self.neg(self.cEmb[left],-negf[:,0])
            righte = self.neg(self.cEmb[right],negf[:,1])
            shape = lefte.shape
            
        elif atype == 1:
            righte = self.neg(self.cEmb[right], negf[:,2])
            shape = righte.shape
            lefte = self.t_norm(self.neg(self.cEmb[left[:,0]],negf[:,0]), self.neg(self.cEmb[left[:,1]],negf[:,1]))

        elif atype == 2:
            lefte = self.neg(self.cEmb[left], negf[:,0])
            shape = lefte.shape
            righte = self.t_norm(self.neg(self.cEmb[right[:,0]],negf[:,1]), self.neg(self.cEmb[right[:,1]],negf[:,2]))

        elif atype == 3:
            lefte = self.neg(self.cEmb[left], negf[:,0])
            shape = lefte.shape
            righte = self.exist(self.rEmb[right[:,0]], self.neg(self.cEmb[right[:,1]],negf[:,1]))

        elif atype == 4:
            lefte = self.neg(self.cEmb[left], negf[:,0])
            shape = lefte.shape
            righte = self.forall(self.rEmb[right[:,0]],self.neg(self.cEmb[right[:,1]], negf[:,1]))
            
            
        elif atype == 5:
            righte = self.neg(self.cEmb[right], negf[:,1])
            shape = righte.shape
            lefte = self.exist(self.rEmb[left[:,0]],self.neg(self.cEmb[left[:,1]], negf[:,0]))

        elif atype == 6:
            righte = self.neg(self.cEmb[right], negf[:,1])
            shape = righte.shape
            lefte = self.forall(self.rEmb[left[:,0]],self.neg(self.cEmb[left[:,1]], negf[:,0]))

        if self.logic_name == "Rule":        
            loss = self.rule_based_loss(lefte, atype, righte, left, right, negf)
        else:
            loss = self.HierarchyLoss(lefte, righte)
          
        return loss