######### This is a collection of useful code fragments ############

######### Try using KNN imputation
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error


knn = KNeighborsRegressor(n_neighbors=20)

val_idxs = np.random.choice(range(len(targets_svd)), 3000, replace=False)
train_mask = ~np.isin(np.arange(len(targets_svd)), val_idxs)
val_mask = ~train_mask
knn = knn.fit(inputs[train_mask], targets_svd[train_mask])
predicted_targets_svd = knn.predict(inputs[val_mask])
print('MSE:', mean_squared_error(predicted_targets_svd, targets_svd[val_mask]))

targets_embedding_as_input = targets_svd
targets_embedding_as_input[val_mask] = predicted_targets_svd
inputs_processed = np.concatenate((targets_embedding_as_input, rna_levels), 1)

train_loader, val_loader = utils.make_loaders(inputs_processed, targets, batch_size=batch_size, num_workers=num_workers, val_idx=val_idxs)

###### Try using NN for imputation

n_components = 48
svd =  TruncatedSVD(n_components=n_components, random_state=1)
targets_svd = svd.fit_transform(targets)

idxs = np.random.choice(range(len(targets_svd)), 40000, replace=False)
mask = ~np.isin(np.arange(len(targets_svd)), idxs)
train_loader, val_loader = utils.make_loaders(inputs_processed[mask], targets_svd[mask], batch_size=batch_size, num_workers=num_workers, val_size=2048)
test_loader = utils.make_loaders(inputs_processed[~mask], batch_size=batch_size, num_workers=num_workers)



# Run MLP_Cite_SVD from Experiment_with_MLPs.ipynb 
    
### Careful
outputs = trainer.transform(test_loader)
outputs = np.concatenate((outputs, rna_levels[~mask]), 1)

train_loader, val_loader = utils.make_loaders(outputs, targets[~mask],
                                              batch_size=batch_size, num_workers=num_workers, val_size=2048)


# Run MLP_Cite_SVD from Experiment_with_MLPs.ipynb  AGAIN!