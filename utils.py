import torch
from torch import nn
import pandas as pd
import numpy as np
from scipy import sparse
import matplotlib.pyplot as plt
from tqdm import tqdm
import inspect
from torch.utils.data import DataLoader
from torch.optim.swa_utils import SWALR


# https://d2l.ai/_modules/d2l/torch.html
class HyperParameters:
    def save_hyperparameters(self, ignore=[]):
        """Defined in :numref:`sec_oo-design`"""
        raise NotImplemented

    def save_hyperparameters(self, ignore=[]):
        """Save function arguments into class attributes.
    
        Defined in :numref:`sec_utils`"""
        frame = inspect.currentframe().f_back
        _, _, _, local_vars = inspect.getargvalues(frame)
        self.hparams = {k:v for k, v in local_vars.items()
                        if k not in set(ignore+['self']) and not k.startswith('_')}
        for k, v in self.hparams.items():
            setattr(self, k, v)
      
            

    
def spearman_cor(y_true, y_pred):
    """Compute the correlation between each rows of the y_true and y_pred tensors.
    Compatible with backpropagation.
    """
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    
    
    y_true_centered = y_true - torch.mean(y_true)
    y_pred_centered = y_pred - torch.mean(y_pred)
    cov_tp = torch.sum(y_true_centered*y_pred_centered)/(y_true.shape[0]-1)
    var_t = torch.sum(y_true_centered**2)/(y_true.shape[0]-1)
    var_p = torch.sum(y_pred_centered**2)/(y_true.shape[0]-1)
    return cov_tp/torch.sqrt(var_t*var_p)
    
    
def plot_progress(trainer):
    epochs = np.arange(len(trainer.train_progress['1mod']))
    validated = len(trainer.val_progress['1mod'][0]) != 0
    cor_calculated = len(trainer.train_progress['2mod_cor'][0]) != 0
    if cor_calculated:
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
    else:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    
    ax1.plot(epochs, np.mean(trainer.train_progress['1mod'], 1), label='Train')
    if validated:
        ax1.plot(epochs, np.mean(trainer.val_progress['1mod'], 1), label='Valiadation')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('AE training, 1 modality')
    ax1.legend()

    ax2.plot(epochs, np.mean(trainer.train_progress['2mod'], 1), label='Train')
    if validated:
        ax2.plot(epochs, np.mean(trainer.val_progress['2mod'], 1), label='Valiadation')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.set_title('AE training, 2 modality')
    ax2.legend()
    
    
    if cor_calculated:
        ax3.plot(epochs, np.mean(trainer.train_progress['2mod_cor'], 1), label='Train')
        if validated:
            ax3.plot(epochs, np.mean(trainer.val_progress['2mod_cor'], 1), label='Valiadation')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Correlation')
        ax3.set_title('AE training, 2 modality correlation')
        ax3.legend()
    plt.show()

def plot_model_analysis(rna_pred, rna_orig):
    
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    
    idxs = np.random.choice(rna_orig.shape[0] * rna_orig.shape[1], size=10**6, replace=False)
    axs[0, 0].scatter(rna_orig.flatten()[idxs],
                rna_pred.flatten()[idxs], alpha=0.2)
    axs[0, 0].set_title('Compare GEX preds with orig')
    axs[0, 0].set_xlabel('Original')
    axs[0, 0].set_ylabel('Predicted')

    axs[0, 1].scatter(rna_orig.mean(0), 
                rna_pred.mean(0), alpha=0.2)
    axs[0, 1].set_title('Compare mean GEX preds with orig')
    axs[0, 1].set_xlabel('Original')
    axs[0, 1].set_ylabel('Predicted')

    axs[1, 0].scatter(rna_orig.mean(0),
                rna_orig.var(0))
    axs[1, 0].set_xlabel('mean')
    axs[1, 0].set_ylabel('variance')
    axs[1, 0].set_title('Distribution of orig GEX')

    axs[1, 1].scatter(rna_pred.mean(0),
                rna_pred.var(0))
    axs[1, 1].set_xlabel('mean')
    axs[1, 1].set_ylabel('variance')
    axs[1, 1].set_title('Distribution of pred GEX data')
    plt.show()

    plt.scatter(rna_orig.var(0), rna_pred.var(0),
                alpha=0.1, c=rna_orig.mean(0))
    plt.xlabel('Orig variance')
    plt.ylabel('Pred variance')
    plt.title('Compare GEX variance between predictions and original')
    plt.colorbar(label='Mean expression')
    plt.show()
    
    #plt.scatter(atac_orig.var(0), atac_pred.var(0),
    #            alpha=0.1, c=atac_orig.mean(0))
    #plt.xlabel('Orig variance')
    #plt.ylabel('Pred variance')
    #plt.title('Compare ATAC variance between predictions and original')
    #plt.colorbar(label='Mean expression')
    #plt.show()
    
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    #means = []
    #means += [(atac_orig == 0).sum(1).mean()]
    #means += [(atac_pred == 0).sum(1).mean()]
    #axs[0].bar([0, 1], means, tick_label=['orig', 'pred'])
    #axs[0].set_title('Zeros in ATAC data')
    #axs[0].set_ylabel('Zeros in sample')

    means = []
    means += [(rna_orig == 0).sum(1).mean()]
    means += [(rna_pred == 0).sum(1).mean()]
    axs[1].bar([0, 1], means, tick_label=['orig', 'pred'])
    axs[1].set_title('Zeros in RNA data')
    axs[1].set_ylabel('Zeros in sample')
    plt.show()


class Trainer(HyperParameters):
    def __init__(self,
                 atac_w: float = 0.5,
                 lr: float = 0.01, 
                 min_lr: float = 1e-4,
                 max_epochs: int = 1000,
                 wd: float = 1e-2,
                 use_one_cycle: bool = True,
                 use_swa: bool = False,
                 swa_start: int = 5,
                 inputs_fn: str = None,
                 targets_fn: str = None,
                 device: str = 'cuda',
                 writer = None,
                 l1_weight = 0,
                 l2_weight = 0,
                 sparsity_beta = 0,
                 sparsity_rho = 0,
                 max_schedule_epoch = None,
                 model_params = {}
                 ):
        self.save_hyperparameters()
        self.use_tensor_board = True if writer is not None else False
        self.max_schedule_epoch = self.max_epochs if self.max_schedule_epoch is None else self.max_schedule_epoch

    
    def calculate_l1_norm(self):
        l1_norm = sum([torch.abs(p).sum() for p in self.model.parameters()])
        return l1_norm
    
    def calculate_l2_norm(self):
        l2_norm = sum([torch.pow(p, 2).sum() for p in self.model.parameters()])
        return l2_norm
    
    def configure_optimizers(self):
        return torch.optim.AdamW(self.model.parameters(), self.lr, weight_decay=self.wd)
    
    def configure_scheduler(self):
        return torch.optim.lr_scheduler.OneCycleLR(self.optim, 
                                                    max_lr=self.lr,
                                                    steps_per_epoch=self.steps_per_epoch,
                                                    epochs=self.max_schedule_epoch)

    def add_model(self, model, model_params={}):
        self.model_template = model
        self.model_params = {**self.model_params,
                             ** model_params,
                             **{'sparsity_beta': self.sparsity_beta,
                                'sparsity_rho': self.sparsity_rho}}
    
    def prepare_model(self):
        self.model = self.model_template(**self.model_params)
        self.model.to(self.device)
        
    def reset_model(self):
        self.model = self.model_template(**self.model_params)
        self.model.to(self.device)
        
    def add_train_loader(self, train_loader):
        self.train_loader = train_loader
        
    def add_val_loader(self, val_loader):
        self.val_loader = val_loader
    
    def fit(self, do_validation=True, calculate_cor=True, subset_train=-1):
        self.steps_per_epoch = self.train_loader.nb_batches
        self.model_params['input_dim'] = self.train_loader.input_dim
        self.model_params['output_dim'] = self.train_loader.output_dim
        self.prepare_model()
        self.optim = self.configure_optimizers()
        if self.use_one_cycle:
            self.scheduler = self.configure_scheduler()
        self.epoch = 0
        if self.use_tensor_board:
            print('Using TensorBoard for output')
        self.do_validation = do_validation
        self.calculate_cor = calculate_cor
        self.train_progress = {'2mod': [], '2mod_cor': []}
        self.val_progress = {'2mod': [], '2mod_cor': []}
        if self.use_swa:
            self.schedule_swa = SWALR(self.optim, swa_lr=self.lr)
        for self.epoch in tqdm(range(self.max_epochs)):
            self.train_batch_idx = 0
            self.val_batch_idx = 0
            self.fit_epoch()
            self.epoch += 1 
        print('Training loss: %.3f' % np.mean(self.train_progress['2mod'][-1]),
              'Training cor: %.3f' % np.mean(self.train_progress['2mod_cor'][-1]),
              'Validation loss: %.3f' % np.mean(self.val_progress['2mod'][-1]),
              'Validation cor: %.3f' % np.mean(self.val_progress['2mod_cor'][-1]), sep='\n')
            
    def fit_epoch(self, calculate_cor=True):
        self.model.train()
        self.train_progress['2mod'].append([])
        self.train_progress['2mod_cor'].append([])
        for batch_inputs, batch_targets in self.train_loader:
            batch_inputs, batch_targets = batch_inputs.to(self.device), batch_targets.to(self.device)
            loss_rna, cor = self.model.training_step(batch_inputs, batch_targets, calculate_cor)
            l1_norm = self.calculate_l1_norm() * self.l1_weight if self.l1_weight else 0
            l2_norm = self.calculate_l2_norm() * self.l2_weight if self.l2_weight else 0
            loss = loss_rna + l1_norm + l2_norm
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()
            if self.epoch < self.max_schedule_epoch and self.use_one_cycle:
                self.scheduler.step()
            elif self.epoch == self.max_schedule_epoch and self.use_one_cycle:
                self.lr = self.min_lr
                self.optim = self.configure_optimizers()
            if self.use_swa and self.epoch>self.swa_start:
                self.schedule_swa.step()
            self.train_progress['2mod'][-1].append(float(loss_rna))
            self.train_progress['2mod_cor'][-1].append(float(cor))
            if self.use_tensor_board:
                n_iter = self.epoch * self.train_loader.nb_batches + self.train_batch_idx
                self.writer.add_scalar('train 2 mod loss', loss_rna, n_iter)
                self.writer.add_scalar('train 2 mod cor', cor, n_iter)  
            self.train_batch_idx += 1
            
        mean_rna_loss= np.mean(self.train_progress['2mod'][-1])
        if self.calculate_cor:
            mean_cor = np.mean(self.train_progress['2mod_cor'][-1])
        
        if not self.use_tensor_board:
            print(f'EPOCH {self.epoch}')
            print('Train 2 modality loss:', mean_rna_loss)
            if self.calculate_cor:
                print('Train 2 modality cor:', mean_cor)
            print('\n')
        
        
        if self.do_validation:
            with torch.no_grad():
                self.model.eval()
                self.val_progress['2mod'].append([])
                self.val_progress['2mod_cor'].append([])
                for batch_inputs, batch_targets in self.val_loader:
                    batch_inputs, batch_targets = batch_inputs.to(self.device), batch_targets.to(self.device)
                    loss_rna, cor = self.model.validation_step(batch_inputs, batch_targets, calculate_cor)
                    self.val_progress['2mod'][-1].append(float(loss_rna))
                    self.val_progress['2mod_cor'][-1].append(float(cor))
                    if self.use_tensor_board:
                        n_iter = self.epoch * self.val_loader.nb_batches + self.val_batch_idx
                        self.writer.add_scalar('val 2 mod loss', loss_rna, n_iter)
                        self.writer.add_scalar('val 2 mod cor', cor, n_iter)  

                    self.val_batch_idx += 1

                mean_rna_loss= np.mean(self.val_progress['2mod'][-1])
                if self.calculate_cor:
                    mean_cor = np.mean(self.val_progress['2mod_cor'][-1])

                if not self.use_tensor_board:
                    print('Validation 2 modality loss:', mean_rna_loss)
                    if self.calculate_cor:
                        print('Validation 2 modality cor:', mean_cor)
                    print('\n')
        
        
    def transform(self, test_loader):
        self.test_loader = test_loader
        self.test_batch_idx = 0
        self.model.eval()
        outputs = []
        for batch_inputs in self.test_loader:
            batch_inputs = batch_inputs.to(self.device)
            batch_outputs = self.model.predict(batch_inputs)
            outputs.append(batch_outputs.to('cpu').detach().numpy())
        return np.concatenate(outputs)
            
            
    def load_model(self, file='trained_model.pt'):
        self.prepare_model()
        self.model.load_state_dict(torch.load(file))
        
    def save_model(self, file='trained_model.pt'):
        torch.save(self.model.state_dict(), file)
    
    def analyze_model(self, loader):    
        self.model.eval()
        rna_pred = []
        rna_orig = []
        for batch_inputs, batch_targets  in loader:
            batch_inputs = batch_inputs.to(self.device)
            rna_recon = self.model.predict(batch_inputs)
            rna_pred.append(rna_recon.to('cpu'))
            rna_orig.append(batch_targets)
            
        rna_pred = torch.cat(rna_pred).detach().numpy()   
        rna_orig = torch.cat(rna_orig).detach().numpy()
        return rna_pred, rna_orig

class Dataset(torch.utils.data.Dataset, HyperParameters):
    def __init__(self, inputs, targets):
        self.save_hyperparameters()

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index):
        return self.inputs[index], self.targets[index]


def make_loaders(
              inputs,
              targets = None,
              batch_size: int = 2048,
              subset_train=-1,
              val_size = 2048*4,
              num_workers = 10
              ):

    if targets is not None:
    
        idx = np.arange(targets.shape[0])
        val_idx = np.random.choice(idx, val_size, replace=False)
        train_idx = idx[~np.isin(idx, val_idx)]

        if subset_train > 0:
            train_idx = np.random.choice(train_idx, subset_train, replace=False)

        nb_batches = len(train_idx) // batch_size
        input_dim = inputs.shape[1]
        output_dim = targets.shape[1]
        
        train_loader = DataLoader(Dataset(inputs[train_idx], targets[train_idx]),
                   shuffle=True, drop_last=True, num_workers=num_workers, batch_size=batch_size)
        train_loader.nb_batches = nb_batches
        train_loader.input_dim = input_dim
        train_loader.output_dim = output_dim
   
        
        nb_batches = len(val_idx) // batch_size
        val_loader = DataLoader(Dataset(inputs[val_idx], targets[val_idx]),
                  shuffle=True, drop_last=True, num_workers=num_workers, batch_size=batch_size)
        val_loader.nb_batches = nb_batches
        val_loader.input_dim = input_dim
        val_loader.output_dim = output_dim
        return train_loader, val_loader
    
    else:
        nb_batches = len(inputs) // batch_size
        loader = DataLoader(inputs, batch_size=batch_size, 
                            shuffle=False, drop_last=False,
                            num_workers=num_workers)
        loader.nb_batches = nb_batches
        return loader

def load_sparse_data(file):
    data = torch.Tensor(sparse.load_npz(file).todense())
    return data

class GaussianHistogram(nn.Module):
    def __init__(self, bins, min, max, sigma):
        super(GaussianHistogram, self).__init__()
        self.bins = bins
        self.min = min
        self.max = max
        self.sigma = sigma
        self.delta = float(max - min) / float(bins)
        self.centers = float(min) + self.delta * (torch.arange(bins, device='cuda:0').float() + 0.5)

    def forward(self, x):
        x = torch.unsqueeze(x, 0) - torch.unsqueeze(self.centers, 1)
        x = torch.exp(-0.5*(x/self.sigma)**2) / (self.sigma * np.sqrt(np.pi*2)) * self.delta
        x = x.sum(dim=1)
        return x
    

class CustomModel(nn.Module, HyperParameters):
    def dist_loss(self, y, y_hat):
        gausshist = GaussianHistogram(bins=20, min=5, max=25, sigma=6)
        dist1 = gausshist(y.flatten())[1:]
        dist2 = gausshist(y_hat.flatten())[1:]
        dist1 += 1
        dist2 += 1
        dist1 /= torch.sum(dist1)
        dist2 /= torch.sum(dist2)
        l = nn.KLDivLoss(reduction="batchmean", log_target=True)(torch.log(dist1), torch.log(dist2))
        return l
    
    def MSE_loss(self, y, y_hat):
        l = nn.MSELoss()(y_hat, y)
        return l
    
    def masked_loss(self, y, y_hat):
        mask = (y_hat<-0.1) | (y_hat>1)
        l = nn.MSELoss()(y_hat[mask], y[mask]) / (torch.sum(mask) + 1) * 0.999
        l += nn.MSELoss()(y_hat[~mask], y[~mask]) / (torch.sum(~mask) + 1) * 0.001
        return l
    
    def cor_loss(self, y, y_hat):
        r = torch.corrcoef(torch.stack((y.flatten(), y_hat.flatten())))[0, 1]
        return -r
    
    def loss(self, y, y_hat):
        l = 0
        l += self.MSE_loss(y, y_hat)
        return l
    
    def kl_divergence(self, rho, rho_hat):
        rho_hat = torch.mean(torch.sigmoid(rho_hat), 1) # sigmoid because we need the probability distributions
        rho = torch.tensor([rho] * len(rho_hat)).to(device)
        return torch.sum(rho * torch.log(rho/rho_hat) + (1 - rho) * torch.log((1 - rho)/(1 - rho_hat)))
    

    def training_step(self, inputs, targets, calculate_cor=True):
        rna_recon = self.forward(inputs)
        loss = self.loss(rna_recon, targets)
        
        if calculate_cor:
            cor = spearman_cor(rna_recon, targets)
        else:
            cor=0
        return loss, cor
    
    def validation_step(self, inputs, targets, calculate_cor=True):
        rna_recon = self.forward(inputs)
        loss = self.loss(rna_recon, targets)
        
        if calculate_cor:
            cor = spearman_cor(rna_recon, targets)
        else:
            cor=0
        return loss, cor
    
    def predict(self, inputs):
        rna_recon = self.forward(inputs)
        return rna_recon
    

def init_weights(m, activation='silu'):
    if isinstance(m, nn.BatchNorm1d):
        nn.init.constant_(m.weight.data, 1)
        nn.init.constant_(m.bias.data, 0)
    
    if activation == 'silu':
        if isinstance(m, nn.Conv1d):
            n = m.kernel_size[0] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2 / n))
            if m.bias is not None:
                nn.init.constant_(m.bias.data, 0)
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.001)
            if m.bias is not None:
                nn.init.constant_(m.bias.data, 0) 
    else:
        if isinstance(m, nn.Linear):
            torch.nn.init.kaiming_uniform_(m.weight)
            m.bias.data.fill_(0.01) 