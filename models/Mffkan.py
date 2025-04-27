# @inproceedings{HSM-TDF,  
#   title={Tongue Diagnosis Framework for Fatty Liver Disease Severity Classification Using Kolmogorov-Arnold Network},  
#   link={https://github.com/MLDMXM2017/HSM-TDF}  
# }  

import numpy as np
import torch
import torch.nn as nn
import sys
import os
script_path = os.path.abspath(__file__)
model_path = os.path.dirname(script_path)
sys.path.insert(0, model_path)
from KANLinear import KANLinear as KAN_Linear
from CNNKAN import CNNKan as CNN_Kan

class MffKan(nn.Module): 
    def __init__(self, num_labels, num_features, drop_rate):
        super().__init__()
        self.num_features = num_features
        self.kan_linears = []

        self.IE = CNN_Kan()
        self.DE = nn.Sequential(KAN_Linear(num_features,        32, # in, out
                                            grid_size=5,spline_order=3, scale_noise=0.01, scale_base=1, scale_spline=1,
                                            base_activation=nn.SiLU, grid_eps=0.02, grid_range=[-1, 1]),
                                  nn.BatchNorm1d(32),
                                  nn.Dropout(drop_rate),
                                  KAN_Linear(32,   128, # in, out
                                            grid_size=5,spline_order=3, scale_noise=0.01, scale_base=1, scale_spline=1,
                                            base_activation=nn.SiLU, grid_eps=0.02, grid_range=[-1, 1]))

        self.FFC = nn.Sequential(nn.BatchNorm1d(640),
                                  nn.Dropout(drop_rate),
                                  KAN_Linear(640, 32, # in, out
                                            grid_size=5,spline_order=3, scale_noise=0.01, scale_base=1, scale_spline=1,
                                            base_activation=nn.SiLU, grid_eps=0.02, grid_range=[-1, 1]),
                                  nn.BatchNorm1d(32),
                                  nn.Dropout(drop_rate),
                                  KAN_Linear(32,        num_labels, # in, out
                                            grid_size=5,spline_order=3, scale_noise=0.01, scale_base=1, scale_spline=1,
                                            base_activation=nn.SiLU, grid_eps=0.02, grid_range=[-1, 1]))
        
        self.M_orient = nn.Parameter(torch.tensor([self.ordinal_multi_expert_labels_generation(num_labels, num_labels - 2)]), requires_grad=False)
        self.M_orient_abs = nn.Parameter(self.M_orient.data.abs(), requires_grad=False)
        self.expert_num = self.M_orient.shape[-1]
        self.MEC = nn.Sequential(nn.BatchNorm1d(640),
                                  nn.Dropout(drop_rate),
                                  KAN_Linear(640, 32, # in, out
                                            grid_size=5,spline_order=3, scale_noise=0.01, scale_base=1, scale_spline=1,
                                            base_activation=nn.SiLU, grid_eps=0.02, grid_range=[-1, 1]),
                                  nn.BatchNorm1d(32),
                                  nn.Dropout(drop_rate),
                                  KAN_Linear(32,        self.expert_num, # in, out
                                            grid_size=5,spline_order=3, scale_noise=0.01, scale_base=1, scale_spline=1,
                                            base_activation=nn.SiLU, grid_eps=0.02, grid_range=[-1, 1]),
                                  nn.Softsign())

        for module in self.DE.modules():
            if isinstance(module, KAN_Linear):
                self.kan_linears.append(module)
        for module in self.FFC.modules():
            if isinstance(module, KAN_Linear):
                self.kan_linears.append(module)
        self.kan_linears.pop(-1)
        for module in self.MEC.modules():
            if isinstance(module, KAN_Linear):
                self.kan_linears.append(module)
        self.kan_linears.pop(-1)

    def ordinal_multi_expert_labels_generation(self, n_L, n_a):
        M0 = -1 * np.ones(((n_L - 1), n_L), dtype=int)
        for i in range(1, n_L):
            M0[i - 1, :i] = 1
        for k in range(1, n_a + 1):
            M = M0.copy() 
            for row in M:
                for j in range(n_L):
                    m_r = row.copy()
                    m_r[j] = 0 
                    if 1 in m_r and -1 in m_r and m_r.tolist() not in M0.tolist():
                        M0 = np.vstack([M0] + [m_r])
        return M0.transpose().tolist()

    def forward(self, X_image, X_physio):
        f_image = self.IE(X_image)
        f_physio = self.DE(X_physio)
        f_fusion = torch.cat((f_image, f_physio), dim=1)
        ffc_out = self.FFC(f_fusion)

        encode = self.MEC(f_fusion) # (batch_size, classifier)
        distance = (encode.unsqueeze(1) * self.M_orient_abs - self.M_orient).pow(2).mean(2) # (batch_size, num_labels)
        mec_out = - distance
        return ffc_out, mec_out, f_fusion, distance, encode
    
    
    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        return sum(
            layer.regularization_loss(regularize_activation, regularize_entropy)
            for layer in self.kan_linears
        )

# define net structure
def get_net(num_features, num_labels, drop_rate):
    net = MffKan(num_labels, num_features, drop_rate)
    return net
