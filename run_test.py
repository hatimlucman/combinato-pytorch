import torch
import h5py
from combinato_algorithm import CombinatoAlgorithm, validate_sorting

# Create the model
model = CombinatoAlgorithm(spc_path=r"C:\Users\hp\Downloads\combinato\spc\cluster.exe")

# Load original extracted spikes for fair sorting comparison
with h5py.File(r"C:\Users\hp\Downloads\combinato\simulation_5\data_simulation_5.h5", 'r') as f:
    original_spikes = torch.tensor(f['pos']['spikes'][:], dtype=torch.float32)

# Sort using original spikes
with torch.no_grad():
    sort_idx, match_idx, distance, artifact_ids = model.sorter(
        original_spikes, 'test_sort_pos', sign='pos')

# Validate against ground truth
sort_result = {
    'pos_sort_idx': sort_idx,
    'pos_artifact_ids': artifact_ids,
}

passed_s, report_s = validate_sorting(
    sort_result,
    r"C:\Users\hp\Downloads\combinato\simulation_5\sort_pos_simple\sort_cat.h5"
)
print(report_s)