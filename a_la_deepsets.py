class Rho(utils.HyperParameters, nn.Module):
    def __init__(self,
                 input_dim: int,
                 output_dim: int = 256,
                 device = 'cpu'):
        super().__init__()
        self.save_hyperparameters()
        
        modules = []
        input_dim = self.input_dim
        output_dim1 = 512
        self.l1 = nn.Linear(input_dim, output_dim1)
        self.bn1 = nn.BatchNorm1d(num_features=output_dim1)          
        self.act1 = nn.SiLU()
        
        input_dim = output_dim1
        output_dim2 = 512 
        self.l2 = nn.Linear(input_dim, output_dim2)
        self.bn2 = nn.BatchNorm1d(num_features=output_dim2)          
        self.act2 = nn.SiLU()
        
        input_dim = output_dim2
        output_dim3 = self.output_dim 
        self.l3 = nn.Linear(input_dim, output_dim3)
        
        self.net = nn.Sequential(self.l1, self.bn1, self.act1,
                                 self.l2, self.bn2, self.act2,
                                 self.l3
                                )
        self.net.apply(utils.init_weights)
        
        
    def forward(self, x):

        out = self.l1(x)
        out = self.bn1(out)
        out1 = self.act1(out)
        
        out = self.l2(out1)
        out = self.bn2(out)
        out2 = self.act2(out)
        
        out = self.l3(out2)

        return out
    
    
class Phi(utils.HyperParameters, nn.Module):
    def __init__(self,
                input_dim:int,
                output_dim:int,
                device = 'cpu'):
        super().__init__()
        self.save_hyperparameters()
        
        modules = []
        input_dim = self.input_dim
        output_dim1 = 1024
        self.l1 = nn.Linear(input_dim, output_dim1)
        self.bn1 = nn.BatchNorm1d(num_features=output_dim1)          
        self.act1 = nn.SiLU()
        
        input_dim = output_dim1
        output_dim2 = 1024 
        self.l2 = nn.Linear(input_dim, output_dim2)
        self.bn2 = nn.BatchNorm1d(num_features=output_dim2)          
        self.act2 = nn.SiLU()
        
        input_dim = output_dim2
        output_dim3 = 1024 
        self.l3 = nn.Linear(input_dim, output_dim3)
        self.bn3 = nn.BatchNorm1d(num_features=output_dim3)          
        self.act3 = nn.SiLU()
        
        input_dim = output_dim3 + output_dim2
        output_dim4 = self.output_dim 
        self.l4 = nn.Linear(input_dim, output_dim4)
        
        self.net = nn.Sequential(self.l1, self.bn1, self.act1,
                                 self.l2, self.bn2, self.act2,
                                 self.l3, self.bn3, self.act3,
                                 self.l4
                                )
        self.net.apply(utils.init_weights)
        
    def forward(self, x):

        out = self.l1(x)
        out = self.bn1(out)
        out1 = self.act1(out)
        
        out = self.l2(out1)
        out = self.bn2(out)
        out2 = self.act2(out)
        
        out = self.l3(out2)
        out = self.bn3(out)
        out3 = self.act3(out)
        
        out = self.l4(torch.concat((out3, out2), 1))
        
        return out
    
    
class DeepSets(utils.CustomModel):
    def __init__(self,
                 input_dim:int,
                 output_dim:int,
                 n_neighbours: int = 10,
                 rho_embedding_size: int = 512,
                device = 'cpu',
                **kwargs):
        super().__init__()
        self.save_hyperparameters()
        
        self.n_neighbours += 1
        
        self.n_features = self.input_dim // self.n_neighbours
        self.rho1 = Rho(self.n_features, self.rho_embedding_size)
        self.rho2 = Rho(self.n_features, self.rho_embedding_size)
        
        self.phi_input_size = self.rho_embedding_size * 2
        self.phi = Phi(self.phi_input_size, self.output_dim) 
        
    def forward(self, x):
        
        out_point = self.rho2(x[:, : self.n_features])
        
        out = []
        for i in range(1, self.n_neighbours):
            out.append(self.rho2(x[:, self.n_features * i: self.n_features * (i+1)]))
        out = torch.stack(out)
        out_neighbours = aggregate_using_norm(out)
        #out_neighbours = torch.max(out, 0).values
                
        out = torch.concat([out_point, out_neighbours], 1)
        
        out = self.phi(out)
        
        return out
        

def aggregate_using_norm(x):
    centered_x = x-torch.mean(x, 0, keepdim=True)
    centered_x_T =  torch.transpose(torch.transpose(centered_x, 0, 1), 1, 2)
    svd = torch.linalg.svd(centered_x_T)
    norms = svd[0][:, :, -1]
    return norms



from sklearn.neighbors import NearestNeighbors


def prepare_data_for_deepsets(inputs_metric, inputs_processed, n_neighbors=n_neighbors):
    neigh = NearestNeighbors(n_neighbors=n_neighbors, radius=10)
    neigh = neigh.fit(inputs_svd)
    kneighbors = neigh.kneighbors(inputs_svd, n_neighbors, return_distance=False)
    input_neighbours = np.apply_along_axis(lambda x: inputs_processed[x].flatten(), 1, kneighbors)
    input_neighbours = np.concatenate((inputs_processed, input_neighbours), 1)
    return input_neighbours