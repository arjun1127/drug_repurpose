import torch

adj = torch.load('models/adjacency.pt', map_location='cpu')
print("Type of adj:", type(adj))
if isinstance(adj, list):
    print("Number of relations:", len(adj))
    for i, a in enumerate(adj):
        print(f"Rel {i} type:", type(a))
        if a.is_sparse:
            print(f"Rel {i} indices shape:", a._indices().shape)
