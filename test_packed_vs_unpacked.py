#!/usr/bin/env python3
"""
Test to compare packed sequences vs individual sequence processing.
This test processes two sequences:
1. First run: Both sequences together in packed mode
2. Second run: Each sequence individually (not packed)
3. Compare the outputs
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
    seqlens = torch.tensor([128, 192], device=device)
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
    initial_states_packed = torch.zeros(num_sequences, nheads, headdim, dstate, dtype=dtype, device=device)

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
    
    out_packed, varlen_states_packed = mamba_chunk_scan_combined(
        x_packed, dt_packed, A, B_packed, C_packed, chunk_size,
        D=None, z=None, dt_bias=dt_bias,
        initial_states=initial_states_packed,
        seq_idx=seq_idx_packed.unsqueeze(0),
        cu_seqlens=cu_seqlens,
        dt_softplus=True,
        return_varlen_states=True
    )
    
    print(f"✓ Packed processing succeeded!")
    print(f"  Output shape: {out_packed.shape}")
    print(f"  Varlen states shape: {varlen_states_packed.shape}")
    
    # ========================================================================
    # RUN 2: Process each sequence SEPARATELY (batch mode, not packed)
    # ========================================================================
    print("\n" + "=" * 80)
    print("RUN 2: Processing each sequence SEPARATELY (batch mode, not packed)")
    print("=" * 80)
    
    out_individual_list = []
    states_individual_list = []
    
    for i in range(num_sequences):
        start_idx = cu_seqlens[i].item()
        end_idx = cu_seqlens[i + 1].item()
        seq_len = end_idx - start_idx
        
        print(f"\n  Processing sequence {i} (length={seq_len})...")
        
        # Extract this sequence's data
        x_seq = x_packed[:, start_idx:end_idx, :, :].contiguous()
        B_seq = B_packed[:, start_idx:end_idx, :, :].contiguous()
        C_seq = C_packed[:, start_idx:end_idx, :, :].contiguous()
        dt_seq = dt_packed[:, start_idx:end_idx, :].contiguous()
        
        # Use this sequence's initial state
        initial_state_seq = initial_states_packed[i:i+1, :, :, :].contiguous()
        
        # Process in standard batch mode (not packed)
        out_seq, final_state_seq = mamba_chunk_scan_combined(
            x_seq, dt_seq, A, B_seq, C_seq, chunk_size,
            D=None, z=None, dt_bias=dt_bias,
            initial_states=initial_state_seq,
            seq_idx=None,
            cu_seqlens=None,
            dt_softplus=True,
            return_final_states=True
        )
        
        out_individual_list.append(out_seq)
        states_individual_list.append(final_state_seq)
        print(f"    ✓ Sequence {i} processed: out_shape={out_seq.shape}, state_shape={final_state_seq.shape}")
    
    # Concatenate results
    out_individual = torch.cat(out_individual_list, dim=1)
    states_individual = torch.cat(states_individual_list, dim=0)
    
    print(f"\n✓ Separate processing completed!")
    print(f"  Concatenated output shape: {out_individual.shape}")
    print(f"  Concatenated states shape: {states_individual.shape}")
    
    # ========================================================================
    # COMPARISON
    # ========================================================================
    print("\n" + "=" * 80)
    print("COMPARISON: Packed vs Separate")
    print("=" * 80)
    
    # Compare outputs
    out_diff = (out_packed - out_individual).abs()
    out_max_diff = out_diff.max().item()
    out_mean_diff = out_diff.mean().item()
    
    # Compare states
    states_diff = (varlen_states_packed - states_individual).abs()
    states_max_diff = states_diff.max().item()
    states_mean_diff = states_diff.mean().item()
    
    print(f"\nOutput Comparison:")
    print(f"  Max absolute difference: {out_max_diff:.6e}")
    print(f"  Mean absolute difference: {out_mean_diff:.6e}")
    
    print(f"\nStates Comparison:")
    print(f"  Max absolute difference: {states_max_diff:.6e}")
    print(f"  Mean absolute difference: {states_mean_diff:.6e}")
    
    # Tolerance check
    rtol, atol = 1e-3, 1e-3
    out_match = torch.allclose(out_packed, out_individual, rtol=rtol, atol=atol)
    states_match = torch.allclose(varlen_states_packed, states_individual, rtol=rtol, atol=atol)
    
    print(f"\nTolerance Check (rtol={rtol}, atol={atol}):")
    print(f"  Outputs match: {out_match}")
    print(f"  States match: {states_match}")
    
    # Detailed per-sequence comparison
    print(f"\nPer-Sequence Output Differences:")
    for i in range(num_sequences):
        start_idx = cu_seqlens[i].item()
        end_idx = cu_seqlens[i + 1].item()
        
        seq_out_packed = out_packed[:, start_idx:end_idx, :, :]
        seq_out_individual = out_individual_list[i]
        
        seq_diff = (seq_out_packed - seq_out_individual).abs()
        seq_max_diff = seq_diff.max().item()
        seq_mean_diff = seq_diff.mean().item()
        
        print(f"  Sequence {i} (length={end_idx-start_idx}):")
        print(f"    Max diff: {seq_max_diff:.6e}, Mean diff: {seq_mean_diff:.6e}")
    
    print(f"\nPer-Sequence State Differences:")
    for i in range(num_sequences):
        state_packed = varlen_states_packed[i:i+1, :, :, :]
        state_individual = states_individual_list[i]
        
        state_diff = (state_packed - state_individual).abs()
        state_max_diff = state_diff.max().item()
        state_mean_diff = state_diff.mean().item()
        
        print(f"  Sequence {i}:")
        print(f"    Max diff: {state_max_diff:.6e}, Mean diff: {state_mean_diff:.6e}")
    
    # Final verdict
    print("\n" + "=" * 80)
    if out_match and states_match:
        print("✅ SUCCESS: Packed and separate processing produce identical results!")
        print("=" * 80)
        return True
    else:
        print("❌ FAILURE: Packed and separate processing produce DIFFERENT results!")
        print("=" * 80)
        if not out_match:
            print(f"   Output mismatch: max diff = {out_max_diff}")
        if not states_match:
            print(f"   States mismatch: max diff = {states_max_diff}")
        return False


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
