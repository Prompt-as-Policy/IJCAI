import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from datetime import datetime
import json
import os
import pickle

# --- Configuration ---
DATA_PATH = "project/testdata.csv"
OUTPUT_DIR = "project/data_processed"
SEED = 42

# --- Constants ---
TIME_SLOTS = {
    'Morning': (6, 12),
    'Afternoon': (12, 18),
    'Evening': (18, 24),
    'Night': (0, 6)
}

def get_time_slot(dt):
    h = dt.hour
    for slot, (start, end) in TIME_SLOTS.items():
        if start <= h < end:
            return slot
    return 'Night' # Default

def process_data():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("Loading data...")
    df = pd.read_csv(DATA_PATH)
    
    # 1. Parse Timestamps
    df['Local_sg_time'] = pd.to_datetime(df['Local_sg_time'])
    df['TimeSlot'] = df['Local_sg_time'].apply(get_time_slot)
    
    # 2. Grid Generation (K-Means)
    print("Generating Grid Cells...")
    unique_venues = df[['VenueId', 'Latitude ', 'Longitude ']].drop_duplicates('VenueId')
    coords = unique_venues[['Latitude ', 'Longitude ']].values
    
    # Dynamic K: sqrt(N_POI / 10)
    n_pois = len(unique_venues)
    if n_pois > 0:
        k = int(np.sqrt(n_pois / 10))
        k = max(2, k) # Minimum 2 clusters
    else:
        k = 1
        
    print(f"Clustering {n_pois} POIs into {k} Grids...")
    kmeans = KMeans(n_clusters=k, random_state=SEED, n_init=10)
    unique_venues['GridLabel'] = kmeans.fit_predict(coords)
    
    # Map back to main df
    grid_map = dict(zip(unique_venues['VenueId'], unique_venues['GridLabel']))
    df['GridID'] = df['VenueId'].map(grid_map)
    
    # 4. Creating Splits (Chronological User Split)
    print("Creating Splits...")
    # Sort users by their first check-in time
    user_start_time = df.groupby('UserId')['Local_sg_time'].min().sort_values()
    all_users = user_start_time.index.tolist()
    
    n_train = int(0.8 * len(all_users))
    train_users = set(all_users[:n_train])
    test_users = set(all_users[n_train:])
    
    train_data = [] 
    test_data = []
    
    # Group by User
    grouped = df.sort_values('Local_sg_time').groupby('UserId')
    
    for uid, group in grouped:
        checkins = group.to_dict('records')
        
        if uid in train_users:
            # Train: Use all history steps
            # Min history length = 3 to start predicting
            if len(checkins) < 2: continue
            
            for i in range(1, len(checkins)):
                # Sample: History [0...i-1], Target [i]
                history = checkins[:i]
                target = checkins[i]
                train_data.append({
                    'user_id': uid,
                    'history': history,
                    'target': target,
                    'split': 'train'
                })
        else:
            # Test: Cold Start Simulation
            # Only use first K (e.g., 3) items as history, predict K+1
            K = 3
            if len(checkins) <= K: continue
            
            history = checkins[:K]
            target = checkins[K] # Predict the immediate next one after cold start
            
            test_data.append({
                'user_id': uid,
                'history': history,
                'target': target,
                'split': 'test_cold_start'
            })
            
    print(f"Train Users: {len(train_users)}, Test Users: {len(test_users)}")
    print(f"Train Samples: {len(train_data)}")
    print(f"Test Samples (Cold Start): {len(test_data)}")
    
    # Save processed data
    with open(f"{OUTPUT_DIR}/train.pkl", 'wb') as f:
        pickle.dump(train_data, f)
    with open(f"{OUTPUT_DIR}/test.pkl", 'wb') as f:
        pickle.dump(test_data, f)
        
    # Save Metadata for KG
    poi_meta = unique_venues.set_index('VenueId')[['Latitude ', 'Longitude ', 'GridLabel']].to_dict('index')
    
    poi_cats = df[['VenueId', 'Category']].drop_duplicates().set_index('VenueId')['Category'].to_dict()
    for vid, cat in poi_cats.items():
        if vid in poi_meta:
            poi_meta[vid]['Category'] = cat
            
    with open(f"{OUTPUT_DIR}/poi_meta.pkl", 'wb') as f:
        pickle.dump(poi_meta, f)

    print("Data processing complete.")

if __name__ == "__main__":
    process_data()
