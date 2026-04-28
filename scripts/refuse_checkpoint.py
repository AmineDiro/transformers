"""Quick re-fuse: gate_proj + up_proj -> gate_up_proj in existing fused checkpoint."""
import json
import shutil
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from tqdm import tqdm


src = Path("/fsx/amine_dirhoussi/Qwen3-30B-A3B-fused")
dst = Path("/fsx/amine_dirhoussi/Qwen3-30B-A3B-fused-v2")
dst.mkdir(exist_ok=True)

with open(src / "model.safetensors.index.json") as f:
    index = json.load(f)

# Copy non-safetensors files
for f in src.iterdir():
    if f.is_file() and not f.name.endswith(".safetensors") and f.name != "model.safetensors.index.json":
        shutil.copy2(f, dst / f.name)

# Process each shard
shard_files = sorted(set(index["weight_map"].values()))
new_weight_map = {}

for shard_name in tqdm(shard_files, desc="Processing shards"):
    tensors = load_file(str(src / shard_name))
    new_tensors = {}

    # Find gate_proj/up_proj pairs in this shard
    gate_keys = {k for k in tensors if ".gate_proj" in k and ".experts." in k}

    processed = set()
    for gk in gate_keys:
        uk = gk.replace(".gate_proj", ".up_proj")
        if uk in tensors:
            fused_key = gk.replace(".gate_proj", ".gate_up_proj")
            new_tensors[fused_key] = torch.cat([tensors[gk], tensors[uk]], dim=1)
            processed.add(gk)
            processed.add(uk)
            print(f"  {fused_key}: {list(new_tensors[fused_key].shape)}")

    for k, v in tensors.items():
        if k not in processed:
            new_tensors[k] = v

    save_file(new_tensors, str(dst / shard_name))
    for k in new_tensors:
        new_weight_map[k] = shard_name

new_index = {"metadata": index.get("metadata", {}), "weight_map": new_weight_map}
with open(dst / "model.safetensors.index.json", "w") as f:
    json.dump(new_index, f, indent=2)
print(f"\nDone! Written to {dst}")
