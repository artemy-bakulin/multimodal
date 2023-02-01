# multimodal

To run NN models create an environment using following commands:

<pre>
conda create -n multiomodal_NN pytorch torchvision torchaudio cudatoolkit=11.6 ipykernel scipy tqdm tensorboard -c pytorch -c conda-forge
conda activate multiomodal_NN
pip install matplotlib
</pre>

To train models I use functions and classes stored in **utils.py**. The main class here is a multifunctional **Trainer** which allows for training models with different sets of hyprparameters, including some unusual ones like whether to use OneCycleLR schedule, whether to use SWA or whether to apply sparsity regularization to target values.


Here is a sample of models I used in the multimodal prediction comptetition:

1.The basic model here is, of course, an MLP. 
In the **Experiment_with_MLPs.ipynb** I provide following models:
* **MLP_Original_Multi**  is a model for predicting gene expression based on untransformed atac data 
* **MLP_SVD_and_Original** predicts uses both original atac data and its svd decomposition 
* **MLP_Cite_SVD** predicts protein levels based on svd decomposition of RNA-seq data 
* **Universal_Model** is a generic class for constructing arbitrary MLPs with a given number of layers and residual connections. It is especially useful for the search of the best architecture during optimization. 

In the bottom of this notebook you can find LRRT which I used for the identification of the optimal learning rate. 
2. As a way of regularization of the learnt MLP embeddings I also tried to add to it an additional reconstruction loss – effectively producing an autoencoder. 

The model is in **Make_AE.ipynb**.

3. I also tried to develop a model with sparse connections between layers.
In **Experiment_with_Sparsity_CITE.ipynb** I leverage information on gene-gene co-occurrence within molecular pathways to cut redundant connections between layers. Here I use module sparselinear to model connectivity. 

4. Inspired by the ideas of invariance and equivariance from geometric learning, I decided to develop a model which aggregates information from atac peaks surrounding each gene to predict its expression and which at the same time uses the same set of weights for the analysis of different genes. The model is based on deepsets architectue and is stored in **a_la_deepsets.py**. 
