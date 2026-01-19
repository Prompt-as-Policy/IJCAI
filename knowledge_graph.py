import networkx as nx
import pandas as pd
import numpy as np
from sklearn.neighbors import BallTree
import pickle
from collections import defaultdict
import math

class KnowledgeGraph:
    def __init__(self, data_dir="project/data_processed"):
        self.data_dir = data_dir
        self.graph = nx.MultiDiGraph()
        self.poi_meta = {}
        self.relations = set()
        
        # Canonical directions for verbalization
        # (h_type, relation, t_type) -> Template
        self.verbalizer = {
            ('User', 'visited', 'POI'): "User visited {}",
            ('User', 'hasProfile', 'Profile'): "User has profile focused on {}",
            ('Profile', 'prefersIntent', 'Intent'): "User profile suggests intent {}",
            ('Profile', 'prefersTime', 'TimeSlot'): "User typically is active in {}",
            ('POI', 'inCategory', 'Category'): "{} is a {}",
            ('POI', 'inGrid', 'Grid'): "{} is located in {}",
            ('POI', 'activeInTime', 'TimeSlot'): "{} is popular during {}",
            ('POI', 'near', 'POI'): "{} is near {}",
            # Inverse maps implicitly handled if checking canonical
        }
        
        # Define inverse relations for graph connectivity
        self.inverse_relations = {
            'visited': 'visited_by',
            'hasProfile': 'profile_of',
            'prefersIntent': 'intent_preferred_by',
            'prefersTime': 'time_preferred_by',
            'inCategory': 'category_contains',
            'inGrid': 'grid_contains',
            'activeInTime': 'time_features',
            'near': 'near' # Symmetric
        }

    def load_data(self):
        print("Loading metadata...")
        with open(f"{self.data_dir}/poi_meta.pkl", 'rb') as f:
            self.poi_meta = pickle.load(f)
            
        with open(f"{self.data_dir}/train.pkl", 'rb') as f:
            self.train_data = pickle.load(f)
            
        # We might also need categories list to create nodes
        self.categories = set()
        for pid, meta in self.poi_meta.items():
            if 'Category' in meta:
                self.categories.add(meta['Category'])

    def build_graph(self):
        print("Building nodes...")
        # 1. Add POI Nodes
        for pid, meta in self.poi_meta.items():
            self.graph.add_node(pid, type='POI', lat=meta['Latitude '], lon=meta['Longitude '])
            
            # Add Category Nodes & Edges
            if 'Category' in meta:
                cat = meta['Category']
                self.graph.add_node(cat, type='Category')
                self.add_bidirectional_edge(pid, cat, 'inCategory')

            # Add Grid Nodes & Edges
            if 'GridLabel' in meta:
                gid = f"Grid_{meta['GridLabel']}"
                self.graph.add_node(gid, type='Grid')
                self.add_bidirectional_edge(pid, gid, 'inGrid')
        
        # 2. Add TimeSlot Nodes
        time_slots = ['Morning', 'Afternoon', 'Evening', 'Night']
        for ts in time_slots:
            self.graph.add_node(ts, type='TimeSlot')

        # 3. Add User Nodes & Interactions (Training Data)
        # Note: We only add 'visited' edges from Training history to avoid leakage?
        # The prompt says "Construct the KG ... to discover candidate POIs".
        # Yes, we use training history to build the user-POI connections.
        print("Adding user history...")
        for sample in self.train_data:
            uid = sample['user_id']
            self.graph.add_node(uid, type='User')
            
            # History edges
            for checkin in sample['history']:
                pid = checkin['VenueId']
                if pid in self.graph: 
                    self.add_bidirectional_edge(uid, pid, 'visited')
                    
                    # activeInTime
                    ts = checkin['TimeSlot'] 
                    if ts in time_slots:
                        self.add_bidirectional_edge(pid, ts, 'activeInTime')

        # 3.b Add Cold Start Users from Test Data
        # We need their nodes to exist for Candidate Discovery (BFS from User)
        # We also need their history (sparse) to exist to jump to POIs.
        print("Adding test users (Cold Start)...")
        # Load test data if not loaded (it's not loaded by default in load_data)
        # We load it here ad-hoc since load_data structure is fixed
        try:
            with open(f"{self.data_dir}/test.pkl", 'rb') as f:
                test_data = pickle.load(f)
            
            for sample in test_data:
                uid = sample['user_id']
                if not self.graph.has_node(uid):
                    self.graph.add_node(uid, type='User')
                    
                    # Add edges for their limited history
                    # Note: We must NOT add the 'target' edge (Leakage!)
                    for checkin in sample['history']:
                        pid = checkin['VenueId']
                        if pid in self.graph:
                            self.add_bidirectional_edge(uid, pid, 'visited')
        except Exception as e:
            print(f"Warning: Could not load test.pkl or add test users: {e}")

        # 4. Add Profile & Intent Nodes (Inferred from *CACHED LLM OUTPUT*)
        print("Constructing User Profiles & Intents...")
        
        # Load Intent Cache
        intent_cache = {}
        try:
             with open(f"{self.data_dir}/intent_cache.pkl", 'rb') as f:
                 intent_cache = pickle.load(f)
             print(f"Loaded {len(intent_cache)} intents for KG construction.")
        except FileNotFoundError:
             print("Warning: intent_cache.pkl not found. Intents will be mocked.")

        for sample in self.train_data:
            uid = sample['user_id']
            # Simple Profile: Top Category
            cats = []
            for h in sample['history']:
                if h['VenueId'] in self.poi_meta:
                    cat = self.poi_meta[h['VenueId']].get('Category')
                    if cat: cats.append(cat)
            
            if cats:
                from collections import Counter
                top_cat = Counter(cats).most_common(1)[0][0]
                
                # Profile Entity
                pid = f"Profile_{uid}"
                self.graph.add_node(pid, type='Profile', label=top_cat)
                self.add_bidirectional_edge(uid, pid, 'hasProfile')
                
                # prefersTime
                times = [h['TimeSlot'] for h in sample['history']]
                if times:
                    top_time = Counter(times).most_common(1)[0][0]
                    self.add_bidirectional_edge(pid, top_time, 'prefersTime')
                
                # prefersIntent (FROM CACHE)
                # Aggregate all inferred intents for this user's history
                user_intents = []
                # Note: intent_cache keys are (uid, time_str)
                # We want the intent of the *user*, so we can aggregate intents from their samples.
                # However, the cache is sparse (only for sample targets).
                # Heuristic: Iterate the user's samples in train_data
                
                # Since we are iterating train_data samples, let's collect intents for this uid
                # This loop processes each sample. We should aggregate outside or per user.
                # But here we are inside loop over samples? 
                # Wait, 'sample' in 'train_data' is one training instance (history + target).
                # A user appears multiple times.
                # We should build the profile once per user.
                pass 

        # Re-organize loop to be User-Centric
        user_samples = defaultdict(list)
        for sample in self.train_data:
            user_samples[sample['user_id']].append(sample)

        for uid, samples in user_samples.items():
            # 1. Profile Label (Top Category across all history)
            all_cats = []
            all_times = []
            user_intent_counts = Counter()

            for s in samples:
                # History Cats
                for h in s['history']:
                    if h['VenueId'] in self.poi_meta:
                         c = self.poi_meta[h['VenueId']].get('Category')
                         if c: all_cats.append(c)
                         all_times.append(h['TimeSlot'])
                
                # Inferred Intent (Target)
                # We use the intent of the target visit as a signal of what the user likes?
                # Or we use the intent of the history visits?
                # offline_intent.py inferred intent for the TARGET info.
                # "User went to P at Time T with Intent I".
                # If we want to say "User prefers Intent I", we should count how often they had that intent.
                ts_str = str(s['target']['Local_sg_time'])
                intent = intent_cache.get((uid, ts_str))
                if intent and intent != "General":
                    user_intent_counts[intent] += 1
            
            if not all_cats: continue
            
            top_cat = Counter(all_cats).most_common(1)[0][0]
            pid = f"Profile_{uid}"
            
            # Add Profile Node if not exists
            if not self.graph.has_node(pid):
                self.graph.add_node(pid, type='Profile', label=top_cat)
                self.add_bidirectional_edge(uid, pid, 'hasProfile')
            
                # prefersTime
                if all_times:
                    top_time = Counter(all_times).most_common(1)[0][0]
                    self.add_bidirectional_edge(pid, top_time, 'prefersTime')
                
                # prefersIntent
                # Top 3 intents?
                for intent, count in user_intent_counts.most_common(3):
                    intent_node = f"Intent_{intent}"
                    self.graph.add_node(intent_node, type='Intent')
                    self.add_bidirectional_edge(pid, intent_node, 'prefersIntent')
            else:
                 # Already added profile (shouldn't happen if we loop by user)
                 pass

    def build_spatial_edges(self):
        print("Building spatial edges (near)...")
        # Extract coordinates
        poi_ids = [n for n, d in self.graph.nodes(data=True) if d.get('type') == 'POI']
        if not poi_ids:
            return
            
        coords = np.array([[self.graph.nodes[p]['lat'], self.graph.nodes[p]['lon']] for p in poi_ids])
        
        # Haversine distance requires radians
        coords_rad = np.radians(coords)
        
        # BallTree for efficient query
        # Earth radius approx 6371 km. 10km threshold.
        # k = radius / earth_radius
        earth_radius_km = 6371.0
        radius_rad = 10.0 / earth_radius_km
        
        tree = BallTree(coords_rad, metric='haversine')
        
        # Query radius
        # query_radius returns indices of neighbors
        indices = tree.query_radius(coords_rad, r=radius_rad)
        
        count = 0
        for i, neighbor_indices in enumerate(indices):
            p1 = poi_ids[i]
            for j in neighbor_indices:
                if i == j: continue # self
                p2 = poi_ids[j]
                # 'near' is symmetric
                self.graph.add_edge(p1, p2, relation='near')
                count += 1
                
        print(f"Added {count} 'near' edges.")
        
    def add_bidirectional_edge(self, u, v, relation):
        # Forward
        self.graph.add_edge(u, v, relation=relation)
        # Backward
        rev_rel = self.inverse_relations.get(relation, f"{relation}_rev")
        self.graph.add_edge(v, u, relation=rev_rel)
        
    def get_neighbors(self, node):
        return self.graph.neighbors(node)
        
    def get_edge_relation(self, u, v):
        # Return ALL relations between u and v
        # NetworkX MultiGraph stores edges as dict of keys
        if self.graph.has_edge(u, v):
            return [data['relation'] for key, data in self.graph[u][v].items()]
        return []

    def is_canonical(self, u, v, relation):
        # Check if (u, v, relation) matches the canonical direction
        # We check our verbalizer keys
        u_type = self.graph.nodes[u].get('type')
        v_type = self.graph.nodes[v].get('type')
        if (u_type, relation, v_type) in self.verbalizer:
            return True
        return False

# Initialize and Build
if __name__ == "__main__":
    kg = KnowledgeGraph()
    # Assuming data exists...
    try:
        kg.load_data()
        kg.build_graph()
        kg.build_spatial_edges()
        
        # Save graph
        print("Saving graph...")
        with open(f"{kg.data_dir}/kg.pkl", 'wb') as f:
            pickle.dump(kg, f)
        print("KG Construction Complete.")
    except Exception as e:
        print(f"Skipping execution due to missing data: {e}")
