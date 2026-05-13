## Part 1 — Prepare both taxonomy versions

**127-cluster centroids (straightforward):**

```python
import numpy as np
import pandas as pd

# Compute centroid for each of the 127 clusters
centroids_127 = (
    cluster_df.groupby('cluster_id')['embedding']
    .apply(lambda x: np.mean(np.stack(x), axis=0))
    .reset_index()
)
centroids_127.columns = ['cluster_id', 'centroid']

# Merge in labels
centroids_127 = centroids_127.merge(
    cluster_df[['cluster_id', 'cluster_label']].drop_duplicates(),
    on='cluster_id'
)
```

**47-cluster centroids (merge first, then recompute):**

```python
# Map original cluster_ids to new labels
cluster_df_merged = cluster_df.merge(
    df_new, on='cluster_id', how='left'
)
# df_new column 2 is the new label — rename it for clarity
cluster_df_merged = cluster_df_merged.rename(
    columns={df_new.columns[1]: 'new_cluster_label'}
)

# Recompute centroids by new label grouping
centroids_47 = (
    cluster_df_merged.groupby('new_cluster_label')['embedding']
    .apply(lambda x: np.mean(np.stack(x), axis=0))
    .reset_index()
)
centroids_47.columns = ['cluster_label', 'centroid']
```

---

## Part 2 — Production emulation function

This is the core tagger you'll run against the full 188k. Build it once and use it for both taxonomy versions:

```python
from sklearn.metrics.pairwise import cosine_similarity

def run_production_emulation(all_strings, all_embeddings, centroids_df, 
                              label_col='cluster_label', threshold=0.45):
    # Stack centroids into matrix for efficient similarity computation
    centroid_matrix = np.stack(centroids_df['centroid'].values)
    labels = centroids_df[label_col].values
    
    results = []
    # Process in batches to avoid memory issues with 188k
    batch_size = 1000
    
    for i in range(0, len(all_embeddings), batch_size):
        batch_embs = np.stack(all_embeddings[i:i+batch_size])
        batch_strings = all_strings[i:i+batch_size]
        
        # Cosine similarity against all centroids
        sims = cosine_similarity(batch_embs, centroid_matrix)
        
        best_idx = sims.argmax(axis=1)
        best_sim = sims.max(axis=1)
        second_sim = np.sort(sims, axis=1)[:, -2]
        
        for j, (string, bidx, bsim, ssim) in enumerate(
            zip(batch_strings, best_idx, best_sim, second_sim)
        ):
            # Diagnosis
            if bsim < 0.30:
                diagnosis = 'genuinely_new'
            elif bsim < threshold and (bsim - ssim) < 0.05:
                diagnosis = 'boundary_ambiguous'
            elif bsim < threshold:
                diagnosis = 'weak_match'
            else:
                diagnosis = 'classified'
                
            results.append({
                'topic': string,
                'assigned_label': labels[bidx] if bsim >= threshold else 'unclassified',
                'confidence': round(float(bsim), 4),
                'diagnosis': diagnosis,
                'second_best_label': labels[np.argsort(sims[j])[-2]],
                'second_best_sim': round(float(ssim), 4)
            })
    
    return pd.DataFrame(results)
```

---

## Run both and compare

```python
# Load full dataset
full_df = pd.read_parquet('embeddings_minilm.parquet')
all_strings = full_df['topic'].tolist()
all_embeddings = full_df['embedding'].tolist()

# Run both
results_127 = run_production_emulation(all_strings, all_embeddings, centroids_127)
results_47 = run_production_emulation(all_strings, all_embeddings, centroids_47)
```

---

## Comparison metrics to look at

Once both are done, run this on each:

```python
def summarize_results(results_df, name):
    total = len(results_df)
    classified = (results_df['diagnosis'] == 'classified').sum()
    unclassified = (results_df['assigned_label'] == 'unclassified').sum()
    ambiguous = (results_df['diagnosis'] == 'boundary_ambiguous').sum()
    
    print(f"\n--- {name} ---")
    print(f"Total strings: {total}")
    print(f"Classified: {classified} ({classified/total:.1%})")
    print(f"Unclassified: {unclassified} ({unclassified/total:.1%})")
    print(f"Boundary ambiguous: {ambiguous} ({ambiguous/total:.1%})")
    print(f"Mean confidence (classified only): "
          f"{results_df[results_df['diagnosis']=='classified']['confidence'].mean():.3f}")
    print(f"Median confidence (classified only): "
          f"{results_df[results_df['diagnosis']=='classified']['confidence'].median():.3f}")
    
    # Distribution of assignments across labels
    print(f"\nTop 5 most assigned labels:")
    print(results_df[results_df['assigned_label'] != 'unclassified']
          ['assigned_label'].value_counts().head())

summarize_results(results_127, '127 clusters')
summarize_results(results_47, '47 clusters')
```

The numbers to focus on when you report back:

- **Unclassified rate** — should be under 8% for a healthy taxonomy
- **Mean confidence** — higher is better; below 0.50 on average suggests centroids are too broad
- **Top assigned labels** — if one label is absorbing 20%+ of all strings, that's your new catch-all problem
