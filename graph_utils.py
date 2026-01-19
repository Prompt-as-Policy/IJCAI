import networkx as nx
from collections import deque
import numpy as np
import math

class CandidateDiscoverer:
    def __init__(self, kg, max_hops=3, distance_threshold=10.0):
        self.kg = kg
        self.max_hops = max_hops
        self.distance_threshold = distance_threshold # km

    def get_candidates(self, user_id, last_poi_id, inferred_intent=None):
        candidates = set()
        visited = set([user_id])
        queue = deque([(user_id, 0)]) 
        
        last_lat, last_lon = None, None
        if last_poi_id and last_poi_id in self.kg.graph.nodes:
            last_lat = self.kg.graph.nodes[last_poi_id].get('lat')
            last_lon = self.kg.graph.nodes[last_poi_id].get('lon')

        while queue:
            curr, depth = queue.popleft()
            if depth >= self.max_hops: continue
                
            neighbors = list(self.kg.get_neighbors(curr))
            
            # Dynamic Intent Injection
            if self.kg.graph.nodes[curr].get('type') == 'Profile' and inferred_intent:
                neighbors.append(inferred_intent)
                
            for n in neighbors:
                if n in visited: continue
                # FIX: Check if node exists (crucial for injected inferred_intent)
                if n not in self.kg.graph.nodes:
                    continue
                
                visited.add(n)
                
                if self.kg.graph.nodes[n].get('type') == 'POI':
                    # Distance Filter
                    if last_lat is not None:
                        n_lat = self.kg.graph.nodes[n].get('lat')
                        n_lon = self.kg.graph.nodes[n].get('lon')
                        # Simple Euclidean approx for speed check (Haversine preferred)
                        # degrees approx to km: 1 deg lat ~ 111km
                        if n_lat and n_lon:
                            dist = math.sqrt((n_lat - last_lat)**2 + (n_lon - last_lon)**2) * 111
                            if dist > self.distance_threshold: continue
                    
                    if n != last_poi_id: 
                        candidates.add(n)
                
                queue.append((n, depth + 1))
        
        return list(candidates)

    def get_candidate_stats(self, candidates, last_poi_id):
        """
        Calculates phi(C_t): [avg_distance, diversity, intent_match, size]
        Returns normalized numpy array.
        """
        if not candidates: return np.zeros(4)
        
        # 1. Avg Distance
        dists = []
        last_node = self.kg.graph.nodes.get(last_poi_id)
        if last_node:
            for c in candidates:
                c_node = self.kg.graph.nodes.get(c)
                if c_node:
                    d = math.sqrt((c_node['lat']-last_node['lat'])**2 + (c_node['lon']-last_node['lon'])**2) * 111
                    dists.append(d)
        avg_dist = np.mean(dists) if dists else 0.0
        
        # 2. Diversity (Category Entropy)
        cats = []
        for c in candidates:
            # Find category neighbor
            for n in self.kg.graph.neighbors(c):
                if self.kg.graph.nodes[n].get('type') == 'Category':
                    cats.append(n)
        
        from collections import Counter
        counts = Counter(cats).values()
        total = sum(counts)
        entropy = 0
        if total > 0:
            probs = [cnt/total for cnt in counts]
            entropy = -sum(p * math.log(p) for p in probs)
            
        # 3. Intent Match (Mocked logic, check if path exists via Intent)
        # For speed, just random score or skip
        intent_match = 0.5 
        
        # 4. Size
        size = len(candidates)
        
        # Normalize (rough heuristics)
        return np.array([
            min(avg_dist / 20.0, 1.0), 
            min(entropy / 3.0, 1.0),
            intent_match,
            min(size / 50.0, 1.0)
        ])

class EvidenceMiner:
    def __init__(self, kg):
        self.kg = kg
        # Create a DiGraph view for path searching (MultiDiGraph not supported by shortest_simple_paths)
        # This collapses parallel edges but preserves connectivity for BFS.
        self.search_graph = nx.DiGraph(kg.graph)

    def mine_evidence(self, user_id, candidate_id, weight_type='balanced'):
        """
        Finds paths and returns sorted rationales.
        weight_type: 'intent_focused', 'distance_focused', 'balanced'
        """
        try:
             # Use the simple DiGraph for path finding
             raw_paths = nx.shortest_simple_paths(self.search_graph, user_id, candidate_id)
             paths = []
             for i, p in enumerate(raw_paths):
                 if i >= 10: break # Limit pool size
                 if len(p) > 5: break 
                 paths.append(p)
        except nx.NetworkXNoPath:
            return []
        except Exception as e:
            # Fallback if any other graph error
            return []
            
        # Verbalize
        scored_rationales = []
        for path in paths:
            text = self.verbalize(path)
            if not text: continue
            
            score = self.score_path(path, weight_type)
            scored_rationales.append((text, score))
            
        # Sort by score desc
        scored_rationales.sort(key=lambda x: x[1], reverse=True)
        return [r[0] for r in scored_rationales]

    def score_path(self, path, weight_type):
        # path is list of nodes
        score = 1.0
        path_nodes = set(path)
        
        # Check node types present
        # Improve score based on node types matching weight_strategy
        types = [self.kg.graph.nodes[n].get('type') for n in path]
        
        if weight_type == 'intent_focused':
            if 'Intent' in types: score += 2.0
            if 'Profile' in types: score += 1.0
        elif weight_type == 'distance_focused':
            if 'Grid' in types: score += 2.0
            # 'near' relation check needed? path is nodes, relations implicit
        
        # Shorter paths generally better
        score += (5 - len(path)) * 0.1
        return score

    def verbalize(self, path):
        segments = []
        for i in range(len(path) - 1):
            u, v = path[i], path[i+1]
            rels = self.kg.get_edge_relation(u, v)
            if not rels: return None
            curr_rel = rels[0]
            
            # Canonical check
            is_rev = curr_rel.endswith('_rev')
            base_rel = curr_rel.replace('_rev', '')
            
            subj, obj = (v, u) if is_rev else (u, v)
            
            # Lookup Template
            s_type = self.kg.graph.nodes[subj].get('type')
            o_type = self.kg.graph.nodes[obj].get('type')
            
            key = (s_type, base_rel, o_type)
            if key in self.kg.verbalizer:
                tmpl = self.kg.verbalizer[key]
                # If reverse, we need passive voice or "visited by"
                # Simple logic: If we are traversing u->v but canonical is v->u (is_rev)
                # We say "u is served by v" or similar.
                # User request: "visited by User"
                
                if is_rev:
                    # Reverse Templates for smooth flow
                    # "u -> v" via rev means "v -> u" is canonical
                    # We are describing u's relationship to v
                    if base_rel == 'visited':
                        filled = f"was visited by {obj}"
                    elif base_rel == 'hasProfile':
                        filled = f"is the profile of {obj}"
                    elif base_rel == 'prefersIntent':
                        filled = f"is the intent preferred by {obj}"
                    elif base_rel == 'prefersTime':
                        filled = f"is the time preferred by {obj}"
                    elif base_rel == 'inCategory':
                        filled = f"includes {obj}"
                    elif base_rel == 'inGrid':
                        filled = f"contains {obj}"
                    elif base_rel == 'activeInTime':
                        filled = f"is a popular time for {obj}"
                    else:
                        filled = f"is related to {obj} ({base_rel})"
                    
                    segments.append(str(filled))
                else:
                    # Canonical forward
                    # Template usually is "User visited {}" -> "visited POI"
                    # We strip the subject from template? 
                    # "User visited {}" -> "visited POI"
                    # Split template?
                    # Simplification: Just output relation + object name
                    segments.append(f"{base_rel} {v}")
            else:
                segments.append(f"{base_rel} {v}")
            
        return " -> ".join(segments)
