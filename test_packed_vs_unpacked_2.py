#!/usr/bin/env python3
"""
Test to compare packed sequences vs individual sequence processing.
1. RUN 1: Both sequences together in packed mode (cu_seqlens, seq_idx, return_varlen_states).
2. RUN 2: Each sequence processed separately at its actual length (no padding), one call per sequence.
3. Compare outputs and final states.

States are comparable only when the reference runs each sequence at actual length;
batch mode with padding would give final_states at end of last chunk (including padding), not at last valid token.
"""

import torch
import torch.nn.functional as F
from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined


def test_packed_vs_unpacked():
    """Compare packed sequence processing vs individual sequence processing."""
    print("Testing packed vs unpacked sequence processing...")
    print("=" * 80)
    
    device = 'cuda'
    dtype = torch.float32
    
    # Two sequences with different lengths
    num_sequences = 2
    seqlens = torch.tensor([100, 192], device=device)
    cu_seqlens = F.pad(seqlens.cumsum(0), (1, 0))
    total_seqlen = seqlens.sum().item()
    
    print(f"\nSequence Configuration:")
    print(f"  Number of sequences: {num_sequences}")
    print(f"  Sequence lengths: {seqlens.tolist()}")
    print(f"  Total sequence length: {total_seqlen}")
    print(f"  cu_seqlens: {cu_seqlens.tolist()}")
    
    # Model dimensions
    batch = 1  # Required for packed sequences
    nheads = 8
    headdim = 64
    ngroups = 4
    dstate = 32
    chunk_size = 64
    
    print(f"\nModel Dimensions:")
    print(f"  nheads: {nheads}, headdim: {headdim}")
    print(f"  ngroups: {ngroups}, dstate: {dstate}")
    print(f"  chunk_size: {chunk_size}")
    
    # Create inputs for packed sequences
    x_packed = torch.randn(batch, total_seqlen, nheads, headdim, dtype=dtype, device=device)
    B_packed = torch.randn(batch, total_seqlen, ngroups, dstate, dtype=dtype, device=device) / 5
    C_packed = torch.randn(batch, total_seqlen, ngroups, dstate, dtype=dtype, device=device) / 5
    dt_packed = torch.randn(batch, total_seqlen, nheads, device=device, dtype=torch.float32) - 4
    A = -0.1 * torch.rand(nheads, device=device)
    dt_bias = torch.randn(nheads, dtype=torch.float32, device=device) * 0.1
    
    # Create per-sequence initial states
    #initial_states_packed = torch.randn(num_sequences, nheads, headdim, dstate, dtype=dtype, device=device) / 10
    initial_states_packed = torch.ones(num_sequences, nheads, headdim, dstate, dtype=dtype, device=device)
    #initial_states_packed = torch.zeros(num_sequences, nheads, headdim, dstate, dtype=dtype, device=device)

    print(f"\nInitial states shape: {initial_states_packed.shape}")
    
    # Create seq_idx for packed sequences
    seq_idx_packed = torch.cat([
        torch.full((s,), i, dtype=torch.int32, device=device) 
        for i, s in enumerate(seqlens)
    ], dim=0)
    
    # ========================================================================
    # RUN 1: Process as PACKED sequences
    # ========================================================================
    print("\n" + "=" * 80)
    print("RUN 1: Processing as PACKED sequences")
    print("=" * 80)
    
    out_packed, states_packed, varlen_states_packed = mamba_chunk_scan_combined(
        x_packed, dt_packed, A, B_packed, C_packed, chunk_size,
        D=None, z=None, dt_bias=dt_bias,
        initial_states=initial_states_packed,
        seq_idx=seq_idx_packed.unsqueeze(0),
        cu_seqlens=cu_seqlens,
        dt_softplus=True,
        return_varlen_states=True,
        return_final_states=True,
    )
    
    print(f"✓ Packed processing succeeded!")
    print(f"  Output shape: {out_packed.shape}")
    print(f"  Varlen states shape: {varlen_states_packed.shape}")
    
    # ========================================================================
    # RUN 2: Process each sequence INDIVIDUALLY at actual length (no padding)
    # ========================================================================
    # Packed varlen_states are final states at the EXACT last valid token per
    # sequence. Batch mode final_states are at the end of the last CHUNK. So we
    # must run each sequence at its actual length (no padding) to get comparable
    # final states; padded batch would give wrong states for shorter sequences.
    print("\n" + "=" * 80)
    print("RUN 2: Processing each sequence INDIVIDUALLY at actual length (no padding)")
    print("=" * 80)
    
    out_individual_list = []
    states_individual_list = []
    
    for i in range(num_sequences):
        start_idx = cu_seqlens[i].item()
        end_idx = cu_seqlens[i + 1].item()
        seq_len = end_idx - start_idx
        
        # Extract this sequence's data at actual length (no padding)
        x_seq = x_packed[:, start_idx:end_idx, :, :].contiguous()
        B_seq = B_packed[:, start_idx:end_idx, :, :].contiguous()
        C_seq = C_packed[:, start_idx:end_idx, :, :].contiguous()
        dt_seq = dt_packed[:, start_idx:end_idx, :].contiguous()
        
        # This sequence's initial state
        initial_state_i = initial_states_packed[i : i + 1, :, :, :].contiguous()
        
        print(f"\n  Sequence {i}: length={seq_len}...")
        out_seq, state_seq = mamba_chunk_scan_combined(
            x_seq, dt_seq, A, B_seq, C_seq, chunk_size,
            D=None, z=None, dt_bias=dt_bias,
            initial_states=initial_state_i,
            seq_idx=None,
            cu_seqlens=None,
            dt_softplus=True,
            return_final_states=True,
        )
        
        out_individual_list.append(out_seq)
        states_individual_list.append(state_seq)
    
    # Concatenate: output along seq dim (batch=1), states along batch dim
    out_individual = torch.cat(out_individual_list, dim=1)
    states_individual = torch.cat(states_individual_list, dim=0)
    
    print(f"\n✓ Batched sematics processing completed!")
    print(f"  output shape: {out_individual.shape}")
    print(f"  states shape: {states_individual.shape}")


    """
    diff_out = out_packed - out_individual
    tol = 1e-5
    different_mask = diff_out.abs() > tol
    if different_mask.any():
        n_diff = different_mask.sum().item()
        print(f"\n  out_packed vs out_individual: {n_diff} different values (tol={tol})")
        indices = different_mask.nonzero(as_tuple=True)
        max_show = 1000
        print(f"  Locations (batch, seq, head, dim) of different values (first {max_show}):")
        for i in range(min(max_show, indices[0].shape[0])):
            loc = tuple(idx[i].item() for idx in indices)
            print(f"    {loc}: packed={out_packed[loc].item():.6f}, individual={out_individual[loc].item():.6f}, diff={diff_out[loc].item():.6e}")
        if n_diff > max_show:
            print(f"    ... and {n_diff - max_show} more")
    else:
        print(f"\n  out_packed and out_individual match (within tol={tol})")
    
    """
    
    print(f"out_packed: {out_packed[0][150][1][4:20].tolist()}")
    print("--------------------------------")
    print(f"out_individual: {out_individual[0][150][1][4:20].tolist()}")

    print("")

    print(f"out_packed: {out_packed[0][101][3][4:20].tolist()}")
    print("--------------------------------")
    print(f"out_individual: {out_individual[0][101][3][4:20].tolist()}")

    print("")
    
    """
    print(f"states_packed: {varlen_states_packed[1][1][1][:10].tolist()}")
    print("--------------------------------")
    print(f"states_individual: {states_individual[1][1][1][:10].tolist()}")

    print("")
    
    print(f"states_packed: {varlen_states_packed[1][1][2][10:20].tolist()}")
    print("--------------------------------")
    print(f"states_individual: {states_individual[1][1][2][10:20].tolist()}")

    diff_out = (out_packed - out_individual)
    diff_states = (varlen_states_packed - states_individual)
   
    #print(f"diff_out: {diff_out}")
    #print(f"diff_states: {diff_states}")
    """
    

if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available. This test requires a GPU.")
        exit(1)
    
    try:
        success = test_packed_vs_unpacked()
        exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ TEST FAILED with exception:")
        print(f"{type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
