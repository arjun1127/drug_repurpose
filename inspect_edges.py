import pickle

with open('models/metadata.pkl', 'rb') as f:
    metadata = pickle.load(f)

edges = metadata.get('edge_list', [])
print("Number of edges:", len(edges))
if len(edges) > 0:
    print("Edge sample:", edges[:5])
    print("Edge tuple length:", len(edges[0]))

