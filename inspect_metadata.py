import pickle
import pprint

with open('models/metadata.pkl', 'rb') as f:
    metadata = pickle.load(f)

print("Metadata Keys:", metadata.keys())
print("Number of drug nodes:", len(metadata.get('drug_nodes', [])))
print("Number of disease nodes:", len(metadata.get('disease_nodes', [])))
print("Drug ID to Name sample:", list(metadata.get('drug_id_to_name', {}).items())[:5])
print("Node types:", metadata.get('node_types', []))

print("Has raw_edges?", 'raw_edges' in metadata)
if 'raw_edges' in metadata:
    print("raw_edges sample:", metadata['raw_edges'][:2])

if 'all_keys' in metadata:
    print("all_keys sample:", metadata['all_keys'][:5])

