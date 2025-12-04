data_split

1. hypergraph input file: use pt_construct.ipynb to convert the hypergraph to the pt file
hypergraph_pretrain_allKG.pt: hypergraph with nodes features pretrained by complete primekg
hypergraph_random_feature.pt：hypergraph with ndoes features initialized randomly
hypergraph_pretrain_noDD.pt: hypergraph with nodes features pretrained by primekg without DDI

2. split data file
splits/DCsplit0.pt ~ splits/DCsplit4.pt: random split data 5 times
make_splits_val.py: run this script to get the split data
sampler.py: three negative sampling methods

3. other files
kg_directed.csv/kg_directed_noDD.csv: process file in pre-training for matching node features to hypergraph nodes
pretrain_node_feature512.pt/pretrain_node_feature512_noDD.pt: nodes features pretrained by complete primekg/primekg without DDI