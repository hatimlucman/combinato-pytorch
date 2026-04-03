"""
verify_clusters.py
==================
Visual verification of spike sorting results.
Shows mean waveforms per cluster from ground truth.

Usage:
    python verify_clusters.py
"""

import h5py
import matplotlib.pyplot as plt
import numpy as np

# Paths
DATAFILE = 'simulation_5/data_simulation_5.h5'
GROUNDTRUTH = 'simulation_5/sort_pos_simple/sort_cat.h5'

# Load spikes
print("Loading spikes...")
with h5py.File(DATAFILE, 'r') as f:
    spikes = f['pos']['spikes'][:]
print(f"  Loaded {spikes.shape[0]} spikes, shape={spikes.shape}")

# Load ground truth assignments
print("Loading ground truth...")
with h5py.File(GROUNDTRUTH, 'r') as f:
    gt_classes = f['classes'][:]
    gt_artifacts = f['artifacts'][:]

# Get artifact cluster IDs
artifact_ids = sorted(gt_artifacts[gt_artifacts[:, 1] > 0, 0].astype(int).tolist())
print(f"  Artifact clusters: {artifact_ids}")

# Summary stats
unique_clusters = np.unique(gt_classes)
print(f"\nCluster summary:")
print(f"  Total spikes: {len(gt_classes)}")
print(f"  Clusters: {unique_clusters}")
print(f"  Unassigned (cluster 0): {(gt_classes == 0).sum()}")

# Per-cluster stats
print(f"\n{'Cluster':<10} {'Count':<10} {'Artifact':<10}")
print("-" * 30)
for cid in unique_clusters:
    count = (gt_classes == cid).sum()
    is_art = 'YES' if cid in artifact_ids else ''
    print(f"{cid:<10} {count:<10} {is_art:<10}")

# Plot mean waveforms
fig, axes = plt.subplots(2, 1, figsize=(12, 8))

# Plot 1: All cluster mean waveforms
ax1 = axes[0]
colors = plt.cm.tab10(np.linspace(0, 1, 10))
for i, cid in enumerate(unique_clusters):
    if cid == 0:
        continue
    cluster_spikes = spikes[gt_classes == cid]
    mean_wave = cluster_spikes.mean(axis=0)
    label = f'Cluster {cid} (n={len(cluster_spikes)})'
    if cid in artifact_ids:
        label += ' [ARTIFACT]'
    ax1.plot(mean_wave, label=label, color=colors[i % 10], 
             linestyle='--' if cid in artifact_ids else '-')

ax1.set_xlabel('Sample')
ax1.set_ylabel('Amplitude')
ax1.set_title('Mean Waveforms per Cluster')
ax1.legend(loc='upper right', fontsize=8)
ax1.grid(True, alpha=0.3)

# Plot 2: Cluster sizes bar chart
ax2 = axes[1]
cluster_ids = [c for c in unique_clusters if c != 0]
cluster_sizes = [(gt_classes == c).sum() for c in cluster_ids]
bar_colors = ['red' if c in artifact_ids else 'steelblue' for c in cluster_ids]
bars = ax2.bar(range(len(cluster_ids)), cluster_sizes, color=bar_colors)
ax2.set_xticks(range(len(cluster_ids)))
ax2.set_xticklabels([f'C{c}' for c in cluster_ids])
ax2.set_xlabel('Cluster ID')
ax2.set_ylabel('Spike Count')
ax2.set_title('Cluster Sizes (red = artifact)')
ax2.grid(True, alpha=0.3, axis='y')

# Add count labels on bars
for bar, count in zip(bars, cluster_sizes):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20,
             str(count), ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('cluster_verification.png', dpi=150)
print(f"\nSaved plot to cluster_verification.png")
plt.show()
