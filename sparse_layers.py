def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01) 
    if isinstance(m, SparseLayer):
        for layer in m.sub_modules:
            torch.nn.init.xavier_uniform_(layer.weight)
            layer.bias.data.fill_(0.01) 
        
class SparseLayer(nn.Module, HyperParameters):
    def __init__(self,
                input_dim: int,
                bin_size: int = 100,
                o_size:int = 100):
        super().__init__()
        self.save_hyperparameters()
        self.n_of_submodules = self.input_dim // self.bin_size 
        self.sub_modules = nn.ModuleList()
        for i in range(self.n_of_submodules - 1):
            self.sub_modules.append(nn.Linear(self.bin_size, self.o_size))
        self.remainder_size = self.input_dim % self.bin_size + self.bin_size
        self.sub_modules.append(nn.Linear(self.remainder_size, self.o_size))
        
    def forward(self, x):
        out = []
        for i in range(self.n_of_submodules-1):
            chunk = x[:, self.bin_size * i: self.bin_size * (i+1)]
            out.append(self.sub_modules[i](chunk))
        chunk = x[:, self.bin_size * (i+1):]
        out.append(self.sub_modules[i+1](chunk))
        out = torch.concat(out, 1)
        return out
            
    


class MLP(nn.Module, HyperParameters):
    def __init__(self,
                input_dim: int,
                output_dim: int
    ):
        super().__init__()
        self.save_hyperparameters()
        #self.n_layers = 3
        
        modules = []
        input_dim = self.input_dim
        output_dim = (input_dim // 10) * 1
        modules.append(SparseLayer(input_dim, bin_size=10, o_size=1))
        modules.append(nn.BatchNorm1d(num_features=output_dim))            
        modules.append(nn.LeakyReLU(0.2))
        
        #input_dim = output_dim
        #output_dim = (input_dim // 50) * 1
        #modules.append(SparseLayer(input_dim, bin_size=50, o_size=1))
        #modules.append(nn.BatchNorm1d(num_features=output_dim))            
        #modules.append(nn.LeakyReLU(0.2))
        
        
        #input_dim = output_dim
        #output_dim = input_dim 
        #modules.append(SparseLayer(input_dim, bin_size=100, o_size=100))
        #modules.append(nn.BatchNorm1d(num_features=output_dim))            
        #modules.append(nn.LeakyReLU(0.2))
        
        #input_dim = output_dim 
        #output_dim = input_dim // 100 + 1
        #modules.append(SparseLayer(input_dim, bin_size=100, o_size=1))
        #modules.append(nn.BatchNorm1d(num_features=output_dim))            
        #modules.append(nn.LeakyReLU(0.2))
        
        input_dim = output_dim
        output_dim = 2000
        modules.append(nn.Dropout(p=0.5))
        modules.append(nn.Linear(input_dim, output_dim))
        modules.append(nn.BatchNorm1d(num_features=output_dim))            
        modules.append(nn.Softplus()) 
        
        input_dim = output_dim 
        output_dim =input_dim
        modules.append(nn.Dropout(p=0.5))
        modules.append(nn.Linear(input_dim, output_dim))
        modules.append(nn.BatchNorm1d(num_features=output_dim))            
        modules.append(nn.Softplus()) 
        
        #input_dim = 2000 
        #output_dim = 2000
        #modules.append(nn.Linear(input_dim, output_dim))
        #modules.append(nn.BatchNorm1d(num_features=output_dim))            
        #modules.append(nn.LeakyReLU(0.2))
        
        input_dim = output_dim
        output_dim = self.output_dim
        modules.append(nn.Dropout(p=0.5))
        modules.append(nn.Linear(input_dim, output_dim))
        modules.append(nn.Softplus())  
            
        self.net = nn.Sequential(*modules) 
        self.net.apply(init_weights)
    
    def forward(self, x):
        out = self.net(x)
        return out
    
    
#############
#Alternative:
    
def init_weights(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01) 
    if isinstance(m, SparseLayer):
        for seq in m.sub_modules:
            for layer in seq:
                if isinstance(layer, nn.Linear):
                    torch.nn.init.xavier_uniform_(layer.weight)
                    layer.bias.data.fill_(0.01) 
        
class SparseLayer(nn.Module, HyperParameters):
    def __init__(self,
                input_dim: int,
                bin_size: int = 100,
                o_size:int = 100):
        super().__init__()
        self.save_hyperparameters()
        self.n_of_submodules = self.input_dim // self.bin_size + 1
        self.sub_modules = nn.ModuleList()
        for i in range(self.n_of_submodules - 1):
            seq = nn.Sequential()
            seq.append(nn.Linear(self.bin_size, 10))
            seq.append(nn.BatchNorm1d(num_features=10))            
            seq.append(nn.LeakyReLU(0.2))
            seq.append(nn.Linear(10, 1))
            seq.append(nn.BatchNorm1d(num_features=1))  
            seq.append(nn.LeakyReLU(0.2))
            self.sub_modules.append(seq)
        self.remainder_size = self.input_dim % self.bin_size
        
        seq = nn.Sequential()
        seq.append(nn.Linear(self.remainder_size, min(self.remainder_size, self.o_size)))
        #seq.append(nn.BatchNorm1d(num_features=min(self.remainder_size, self.o_size)))            
        seq.append(nn.LeakyReLU(0.2))
        self.sub_modules.append(seq)
        
    def forward(self, x):
        out = []
        for i in range(self.n_of_submodules-1):
            chunk = x[:, self.bin_size * i: self.bin_size * (i+1)]
            out.append(self.sub_modules[i](chunk))
        chunk = x[:, self.bin_size * (i+1):]
        out.append(self.sub_modules[i+1](chunk))
        out = torch.concat(out, 1)
        return out
