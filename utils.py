import torch
from torch import nn
import pandas as pd
import numpy as np
from scipy import sparse
import matplotlib.pyplot as plt
from tqdm import tqdm
import inspect


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
      
            
class DataLoader(torch.utils.data.DataLoader):

    def __init__(self, train_inputs, train_targets, train_idx=None, 
                 *,
                batch_size=512, shuffle=False, drop_last=False, device='cuda'):
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.device = device
        
        self.train_inputs = train_inputs
        self.dataset = np.arange(self.train_inputs.shape[0] // batch_size * batch_size) ### very dumb crutch to use with 
        self.train_targets = train_targets
        self.n_input_features = train_inputs.shape[1]
        self.n_target_features = train_targets.shape[1]
        
        if self.train_inputs.shape[0] < 512 * 4 and False:
            self.data_in_memory = True
            self.train_inputs = torch.from_numpy(self.train_inputs.todense()).to(self.device)
            self.train_targets = torch.from_numpy(self.train_targets.todense()).to(self.device)
        else:
            self.data_in_memory = False
        
        self.train_idx = train_idx
        
        self.nb_examples = len(self.train_idx) if self.train_idx is not None else train_inputs.shape[0]
        
        self.nb_batches = self.nb_examples // batch_size
        if not drop_last and not self.nb_examples%batch_size==0:
            self.nb_batches +=1
        
    def __iter__(self):
        if self.data_in_memory:
            return self.train_inputs, self.train_targets
        
        if self.shuffle:
            shuffled_idx = torch.randperm(self.nb_examples)
            if self.train_idx is not None:
                idx_array = self.train_idx[shuffled_idx]
            else:
                idx_array = shuffled_idx
        else:
            if self.train_idx is not None:
                idx_array = self.train_idx
            else:
                idx_array = None
            
        for i in range(self.nb_batches):
            slc = slice(i*self.batch_size, (i+1)*self.batch_size)
            if idx_array is None:
                inp_batch = self.train_inputs[i*self.batch_size: (i+1)*self.batch_size]
                tgt_batch = self.train_targets[i*self.batch_size: (i+1)*self.batch_size]
            else:
                idx_batch = idx_array[slc]
                inp_batch = self.train_inputs[idx_batch]
                tgt_batch = self.train_targets[idx_batch]
                
            inp_batch = torch.from_numpy(inp_batch.todense()).to(self.device)
            tgt_batch = torch.from_numpy(tgt_batch.todense()).to(self.device)
            yield inp_batch, tgt_batch
            
            
    def __len__(self):
        return self.nb_batches
    
    
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

def plot_model_analysis(atac_pred, rna_pred, atac_orig, rna_orig):
    
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
    
    plt.scatter(atac_orig.var(0), atac_pred.var(0),
                alpha=0.1, c=atac_orig.mean(0))
    plt.xlabel('Orig variance')
    plt.ylabel('Pred variance')
    plt.title('Compare ATAC variance between predictions and original')
    plt.colorbar(label='Mean expression')
    plt.show()
    
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    means = []
    means += [(atac_orig == 0).sum(1).mean()]
    means += [(atac_pred == 0).sum(1).mean()]
    axs[0].bar([0, 1], means, tick_label=['orig', 'pred'])
    axs[0].set_title('Zeros in ATAC data')
    axs[0].set_ylabel('Zeros in sample')

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
                 max_epochs: int = 1000,
                 wd: float = 10e-2,
                 batch_size: int = 2048, 
                 use_schedule: bool = True,
                 inputs_fn: str = None,
                 targets_fn: str = None,
                 device: str = 'cuda',
                 writer = None
                 ):
        self.save_hyperparameters()
        self.use_tensor_board = True if writer is not None else False

        
    def prepare_model(self, model):
        model = model(self.train_loader.n_input_features,
              self.train_loader.n_target_features,
              **self.model_params
              )
        model.to(self.device)
        return model
    
    def calculate_l1_norm(self):
        l1_norm = sum([torch.abs(p).sum() for p in self.model.parameters()])
        return l1_norm
    
    def configure_optimizers(self):
        return torch.optim.AdamW(self.model.parameters(), self.lr, weight_decay=self.wd)
    
    def configure_scheduler(self):
        return torch.optim.lr_scheduler.OneCycleLR(self.optim, 
                                                    max_lr=self.lr,
                                                    steps_per_epoch=self.steps_per_epoch,
                                                    epochs=self.max_epochs)

    def fit(self, model, model_params={}, do_validation=True, calculate_cor=True, subset_train=-1):
        self.train_loader, self.val_loader = self.load_data('train', subset_train=subset_train, batch_size=self.batch_size)
        self.steps_per_epoch = self.train_loader.nb_batches
        self.model_params = model_params
        self.model = self.prepare_model(model)
        self.optim = self.configure_optimizers()
        if self.use_schedule:
            self.scheduler = self.configure_scheduler()
        self.epoch = 0
        if self.use_tensor_board:
            print('Using TensorBoard for output')
        self.do_validation = do_validation
        self.calculate_cor = calculate_cor
        self.train_progress = {'1mod': [], '2mod': [], '2mod_cor': []}
        self.val_progress = {'1mod': [], '2mod': [], '2mod_cor': []}
        for self.epoch in tqdm(range(self.max_epochs)):
            self.train_batch_idx = 0
            self.val_batch_idx = 0
            self.fit_epoch()
            self.epoch += 1 
        print('Training loss: %.2f' % np.mean(self.train_progress['2mod'][-1]),
              'Training cor: %.2f' % np.mean(self.train_progress['2mod_cor'][-1]),
              'Validation loss: %.2f' % np.mean(self.val_progress['2mod'][-1]),
              'Validation cor: %.2f' % np.mean(self.val_progress['2mod_cor'][-1]), sep='\n')
            
    def fit_epoch(self, calculate_cor=True):
        self.model.train()
        self.train_progress['1mod'].append([])
        self.train_progress['2mod'].append([])
        self.train_progress['2mod_cor'].append([])
        for batch_inputs, batch_targets in self.train_loader:
            loss_atac, loss_rna, cor = self.model.training_step(batch_inputs, batch_targets, calculate_cor)
            loss = self.atac_w * loss_atac + (1-self.atac_w) * loss_rna # + self.calculate_l1_norm()
            self.optim.zero_grad()
            loss.backward()
            self.optim.step()            
            if self.use_schedule:
                self.scheduler.step()
            self.train_progress['1mod'][-1].append(float(loss_atac))
            self.train_progress['2mod'][-1].append(float(loss_rna))
            self.train_progress['2mod_cor'][-1].append(float(cor))
            if self.use_tensor_board:
                n_iter = self.epoch * len(self.train_loader) + self.train_batch_idx
                self.writer.add_scalar('train 1 mod loss', loss_atac, n_iter)
                self.writer.add_scalar('train 2 mod loss', loss_rna, n_iter)
                self.writer.add_scalar('train 2 mod cor', cor, n_iter)  
            self.train_batch_idx += 1
            
        mean_atac_loss = np.mean(self.train_progress['1mod'][-1])
        mean_rna_loss= np.mean(self.train_progress['2mod'][-1])
        if self.calculate_cor:
            mean_cor = np.mean(self.train_progress['2mod_cor'][-1])
        
        if not self.use_tensor_board:
            print(f'EPOCH {self.epoch}')
            print('Train 1 modality loss:', mean_atac_loss)
            print('Train 2 modality loss:', mean_rna_loss)
            if self.calculate_cor:
                print('Train 2 modality cor:', mean_cor)
            print('\n')
        
        
        if self.do_validation:
            with torch.no_grad():
                self.model.eval()
                self.val_progress['1mod'].append([])
                self.val_progress['2mod'].append([])
                self.val_progress['2mod_cor'].append([])
                for batch_inputs, batch_targets in self.val_loader:
                    loss_atac, loss_rna, cor = self.model.validation_step(batch_inputs, batch_targets, calculate_cor)
                    self.val_progress['1mod'][-1].append(float(loss_atac))
                    self.val_progress['2mod'][-1].append(float(loss_rna))
                    self.val_progress['2mod_cor'][-1].append(float(cor))
                    if self.use_tensor_board:
                        n_iter = self.epoch * len(self.val_loader) + self.val_batch_idx
                        self.writer.add_scalar('val 1 mod loss', loss_atac, n_iter)
                        self.writer.add_scalar('val 2 mod loss', loss_rna, n_iter)
                        self.writer.add_scalar('val 2 mod cor', cor, n_iter)  

                    self.val_batch_idx += 1

                mean_atac_loss = np.mean(self.val_progress['1mod'][-1])
                mean_rna_loss= np.mean(self.val_progress['2mod'][-1])
                if self.calculate_cor:
                    mean_cor = np.mean(self.val_progress['2mod_cor'][-1])

                if not self.use_tensor_board:
                    print('Validation 1 modality loss:', mean_atac_loss)
                    print('Validation 2 modality loss:', mean_rna_loss)
                    if self.calculate_cor:
                        print('Validation 2 modality cor:', mean_cor)
                    print('\n')
        
        
    def test_model(self, test_fn):
        self.test_fn = test_fn
        self.test_loader = self.load_data('test')
        self.test_batch_idx = 0
        self.model.eval()
        outputs = []
        for batch_inputs, batch_targets in self.test_loader:
            batch_outputs = self.model(batch_inputs)[1]
            outputs.append(batch_outputs.to('cpu').detach().numpy())
        return np.concatenate(outputs)
            
            
    def load_model(self, model, file='trained_model.pt', model_params={}):
        self.model_params = model_params
        self.model = self.prepare_model(model)
        self.model.load_state_dict(torch.load(file))
        
    def save_model(self, file='trained_model.pt'):
        torch.save(self.model.state_dict(), file)
    
    def analyze_model(self, model, dset='train'):    
        if dset=='train':
            loader = self.train_loader
        elif dset=='val':
            loader = self.val_loader
        else:
            self.test_loader = self.load_data('test')
        self.model.eval()
        atac_pred = []
        rna_pred = []
        atac_orig = []
        rna_orig = []
        for batch_inputs, batch_targets  in loader:
            atac_recon, rna_recon = self.model.predict(batch_inputs)
            atac_pred.append(atac_recon.to('cpu'))
            rna_pred.append(rna_recon.to('cpu'))
            atac_orig.append(batch_inputs.to('cpu'))
            rna_orig.append(batch_targets.to('cpu'))
            
        atac_pred = torch.cat(atac_pred).detach().numpy()
        rna_pred = torch.cat(rna_pred).detach().numpy()
        atac_orig = torch.cat(atac_orig).detach().numpy()
        rna_orig = torch.cat(rna_orig).detach().numpy()
        
        return atac_pred, rna_pred, atac_orig, rna_orig

    def load_data(self,
                  dset: str = 'train',
                  cv_split: bool = True,
                  batch_size: int = 2048,
                  subset_train=-1,
                  val_size = 2048*4
                  ):
        if dset == 'train':
            inputs_fn = self.inputs_fn
            targets_fn = self.targets_fn
        elif dset == 'test':
            inputs_fn = self.test_fn
            targets_fn = self.test_fn
        inputs = sparse.load_npz(inputs_fn)
        targets = sparse.load_npz(targets_fn)

        if cv_split and dset == 'train':
            idx = np.arange(targets.shape[0])
            val_idx = np.random.choice(idx, val_size, replace=False)
            train_idx = idx[~np.isin(idx, val_idx)]
            
            if subset_train > 0:
                train_idx = np.random.choice(train_idx, subset_train, replace=False)
            
            train_loader = DataLoader(inputs, targets, 
                                         train_idx=train_idx,
                                         batch_size=batch_size,
                                         drop_last=True,
                                         shuffle=True,
                                         device=self.device)
            val_loader = DataLoader(inputs, targets,
                                       train_idx=val_idx,
                                       batch_size=batch_size,
                                       drop_last=True,
                                       shuffle=True,
                                       device=self.device)
            return train_loader, val_loader
        else:
            loader = DataLoader(inputs, targets, 
                                         batch_size=batch_size,
                                         drop_last=False,
                                         device=self.device)
            return loader