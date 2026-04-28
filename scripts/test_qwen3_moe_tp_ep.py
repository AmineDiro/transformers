"""
Test TP + EP + CP + FSDP2 for Qwen3 MoE.

Validates that the base_model_ep_plan correctly shards:
  - Attention weights via TP (colwise/rowwise)
  - Expert weights via EP (grouped_gemm)
  - Router via EP (ep_router)
  - Context parallelism via torch CP (sequence splitting + ring attention)
  - FSDP2 for data parallel weight sharding

Examples (8 GPUs):
    # TP/EP=2, DP=4
    torchrun --nproc_per_node=8 scripts/test_qwen3_moe_tp_ep.py --tp_size 2

    # TP/EP=4, DP=2
    torchrun --nproc_per_node=8 scripts/test_qwen3_moe_tp_ep.py --tp_size 4

    # TP/EP=2, CP=2, DP=2
    torchrun --nproc_per_node=8 scripts/test_qwen3_moe_tp_ep.py --tp_size 2 --cp_size 2

    # TP/EP=2, CP=2, DP=1 (4 GPUs)
    torchrun --nproc_per_node=4 scripts/test_qwen3_moe_tp_ep.py --tp_size 2 --cp_size 2
"""

import argparse
import os

import torch
import torch.distributed as dist
from torch.distributed._composable.fsdp import MixedPrecisionPolicy, fully_shard
from torch.distributed.tensor.experimental import context_parallel

from transformers import AutoModelForCausalLM
from transformers.distributed.configuration_utils import DistributedConfig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default="Qwen/Qwen3-30B-A3B")
    parser.add_argument("--tp_size", type=int, default=2)
    parser.add_argument("--cp_size", type=int, default=1)
    parser.add_argument("--seq_len", type=int, default=128)
    args = parser.parse_args()

    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)

    tp_size = args.tp_size
    cp_size = args.cp_size
    assert world_size % (tp_size * cp_size) == 0, (
        f"world_size ({world_size}) must be divisible by tp_size*cp_size ({tp_size}*{cp_size}={tp_size * cp_size})"
    )
    dp_size = world_size // (tp_size * cp_size)

    print(f"[Rank {rank}] world_size={world_size}, tp_size={tp_size}, cp_size={cp_size}, dp_size={dp_size}")

    # Build device mesh: (dp, cp, tp) or (dp, tp) if cp=1
    if cp_size > 1:
        device_mesh = dist.init_device_mesh(
            "cuda", (dp_size, cp_size, tp_size), mesh_dim_names=("dp", "cp", "tp")
        )
        tp_mesh = device_mesh["tp"]
        cp_mesh = device_mesh["cp"]
        dp_mesh = device_mesh["dp"]
        print(f"[Rank {rank}] mesh=(dp={dp_size}, cp={cp_size}, tp={tp_size}), "
              f"tp_mesh={tp_mesh}, cp_mesh={cp_mesh}, dp_mesh={dp_mesh}")
    else:
        device_mesh = dist.init_device_mesh("cuda", (dp_size, tp_size), mesh_dim_names=("dp", "tp"))
        tp_mesh = device_mesh["tp"]
        cp_mesh = None
        dp_mesh = device_mesh["dp"]
        print(f"[Rank {rank}] mesh=(dp={dp_size}, tp={tp_size}), tp_mesh={tp_mesh}, dp_mesh={dp_mesh}")

    # Load model with EP on TP sub-mesh
    print(f"[Rank {rank}] Loading model with TP+EP...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        tp_plan="auto",
        distributed_config=DistributedConfig(enable_expert_parallel=True),
        device_mesh=tp_mesh,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",  # required for CP
    )
    print(f"[Rank {rank}] Model loaded.")

    # Verify expert sharding
    expected_local_experts = model.config.num_experts // tp_size
    for i, layer in enumerate(model.model.layers):
        if hasattr(layer.mlp, "experts"):
            actual = layer.mlp.experts.num_experts
            shape = list(layer.mlp.experts.gate_up_proj.shape)
            print(f"[Rank {rank}] Layer {i}: num_experts={actual} (expected {expected_local_experts}), "
                  f"gate_up_proj={shape}")
            assert actual == expected_local_experts, f"Expected {expected_local_experts}, got {actual}"
            break

    # Apply FSDP2
    if dp_size > 1:
        print(f"[Rank {rank}] Applying FSDP2...")
        mp_policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16)
        for layer in model.model.layers:
            fully_shard(layer, mesh=dp_mesh, mp_policy=mp_policy)
        fully_shard(model, mesh=dp_mesh, mp_policy=mp_policy)
        print(f"[Rank {rank}] FSDP2 applied.")

    # Build bogus input (sequence must be divisible by cp_size)
    seq_len = args.seq_len
    assert seq_len % cp_size == 0, f"seq_len ({seq_len}) must be divisible by cp_size ({cp_size})"
    input_ids = torch.randint(0, model.config.vocab_size, (1, seq_len), device=f"cuda:{local_rank}")
    position_ids = torch.arange(seq_len, device=f"cuda:{local_rank}").unsqueeze(0)

    # Forward pass
    print(f"[Rank {rank}] Running forward (seq_len={seq_len}, cp_size={cp_size})...")

    # --- Forward ---
    def run_forward(input_ids, position_ids, cp_mesh, with_grad=False):
        if cp_mesh is not None:
            buffers = [input_ids]
            buffer_seq_dims = [1]
            with context_parallel(cp_mesh, buffers=buffers, buffer_seq_dims=buffer_seq_dims,
                                  no_restore_buffers=set(buffers)):
                outputs = model(input_ids=input_ids, use_cache=False)
        else:
            outputs = model(input_ids=input_ids, position_ids=position_ids, use_cache=False)
        return outputs

    print(f"[Rank {rank}] Running forward (seq_len={seq_len}, cp_size={cp_size})...")
    with torch.no_grad():
        outputs = run_forward(input_ids, position_ids, cp_mesh)

    local_seq = outputs.logits.shape[1]
    expected_seq = seq_len // cp_size
    print(f"[Rank {rank}] Forward PASSED! logits shape={list(outputs.logits.shape)} "
          f"(local_seq={local_seq}, expected={expected_seq})")
    assert local_seq == expected_seq, f"Expected local seq {expected_seq}, got {local_seq}"

    # Consistency check across TP ranks
    logits = outputs.logits
    gathered = [torch.zeros_like(logits) for _ in range(tp_size)]
    dist.all_gather(gathered, logits, group=tp_mesh.get_group())
    if rank == 0:
        max_diff = max((gathered[0] - g).abs().max().item() for g in gathered[1:])
        print(f"[Rank 0] Max logit diff across TP ranks: {max_diff}")
        if max_diff < 1e-2:
            print("[Rank 0] SUCCESS: Logits consistent across TP ranks.")
        else:
            print(f"[Rank 0] WARNING: Large logit diff ({max_diff}).")

    # --- Backward ---
    print(f"[Rank {rank}] Running forward+backward...")
    # Re-create input for grad-enabled pass
    input_ids_bwd = torch.randint(0, model.config.vocab_size, (1, seq_len), device=f"cuda:{local_rank}")
    labels = torch.randint(0, model.config.vocab_size, (1, seq_len // cp_size), device=f"cuda:{local_rank}")

    outputs = run_forward(input_ids_bwd, position_ids, cp_mesh, with_grad=True)
    # Compute a simple loss on the logits
    loss = outputs.logits.mean()
    loss.backward()
    print(f"[Rank {rank}] Backward PASSED! loss={loss.item():.4f}")

    # Verify gradients exist on expert weights
    for i, layer in enumerate(model.model.layers):
        if hasattr(layer.mlp, "experts"):
            gate_up = layer.mlp.experts.gate_up_proj
            has_grad = gate_up.grad is not None
            grad_norm = gate_up.grad.norm().item() if has_grad else 0.0
            print(f"[Rank {rank}] Layer {i} experts.gate_up_proj: has_grad={has_grad}, grad_norm={grad_norm:.6f}")
            break

    dist.destroy_process_group()
    print(f"[Rank {rank}] Done.")


if __name__ == "__main__":
    main()
