# HyperDC: A Non-uniform Hypergraph Framework for Dual- and Higher-order Drug Combination Recommendation across Diverse Complex Diseases

<img width="11338" height="6802" alt="graphical abstract" src="https://github.com/user-attachments/assets/a988deb7-7403-4002-bdbe-924ed650c2bb" />


## Overview:

- `HyperDC_model/` — Main framework directory, containing the core training and prediction scripts. Final prediction and result generation can be performed using `predict_final.ipynb`.
- `data_split/` — Data splitting and preprocessing directory. Includes pre-trained node embeddings and the prepared hypergraph input files in `.pt` format.
- `hypergraph_construct/` — Directory containing the fully constructed final hypergraph.
- `disease_map/` — Code and datasets for evaluating disease-term matching performance across different BERT models.

## Requirements:

python=3.8.18  
torch==2.0.1  
torchvision==0.15.2  
torchaudio==2.0.2  
scikit-learn==1.3.2  
torchmetrics==1.5.2  
tqdm==4.66.5  
dgl-cu117==0.9.1post1  
pandas==1.4.0  
transformers==4.46.3  
matplotlib==3.7.2  
openpyxl==3.1.5  

## Prediction:

If you want to directly run predictions using the pre-constructed models from the paper, first obtain the target disease or drug `hypergraph_idx` from `node_idx_map.csv` located in the `hypergraph_construct` directory. Specify these selected node indices in the `fixed_indices = []` list inside `predict_final.ipynb`. You may freely choose how many nodes to fix (e.g., specifying only a disease, or both a disease and an anchor drug). The script will automatically pair every other drug node (`hypergraph_idx` from 0 to 2773) with the fixed nodes, compute the combination scores, and generate a prediction file with results ranked from highest to lowest score.
