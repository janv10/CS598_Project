# Reproducibility Study for GAT and ResidualGAT on Citation Network Datasets

1. Citation to the paper
Petar Veličković, Guillem Cucurull, Arantxa Casanova, Adriana Romero, Pietro Liò, & Yoshua Bengio. (2018). Graph Attention Networks. Link: https://arxiv.org/abs/1710.10903

2. Link to the original paper's repository
GitHub Link: https://github.com/PetarV-/GAT

3. Dependencies
    - train.py:
        - sklearn.metrics (imported f1_score, precision_score, recall_score)
    - models.py:
        - torch.nn.functional (imported as F)
        - torch_geometric.nn (imported GATConv)
        - torch.nn (imported LayerNorm)
    - datasets.py:
        - torch_geometric.transforms (imported as T)
        - torch_geometric.datasets (imported Planetoid)
    - main.py:
        - torch_geometric.transforms (imported as T)
        - torch_geometric.datasets (imported Planetoid)

4. Data Download Instructions
Download the source code from GitHub. 
    - Code -> "Download as ZIP"

## Execute Code
5. Preprocessing code + command
6. Training code + command
7. Pretrained model
    - To run the python code, use command "python3 main.py" within the source folder.
    - To run the Jupyter Notebook for visualization, use command "jupyter notebook" within the source folder. 


8. Table of results
    - Results are located in "data_output" folder

