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
    # RUN 2: Process each sequence SEPARATELY (batch mode, not packed)
    # ========================================================================
    print("\n" + "=" * 80)
    print("RUN 2: Processing each sequence SEPARATELY (batch mode, not packed)")
    print("=" * 80)
    
    # Prepare all sequences as a regular batch (pad to max length)
    max_seqlen = seqlens.max().item()
    x_batch_list = []
    B_batch_list = []
    C_batch_list = []
    dt_batch_list = []
    
    print(f"\n  Preparing batch with max_seqlen={max_seqlen}...")
    for i in range(num_sequences):
        start_idx = cu_seqlens[i].item()
        end_idx = cu_seqlens[i + 1].item()
        seq_len = end_idx - start_idx
        
        # Extract this sequence's data
        x_seq = x_packed[:, start_idx:end_idx, :, :]
        B_seq = B_packed[:, start_idx:end_idx, :, :]
        C_seq = C_packed[:, start_idx:end_idx, :, :]
        dt_seq = dt_packed[:, start_idx:end_idx, :]
        
        # Pad to max length if necessary
        if seq_len < max_seqlen:
            pad_len = max_seqlen - seq_len
            x_seq = F.pad(x_seq, (0, 0, 0, 0, 0, pad_len))
            B_seq = F.pad(B_seq, (0, 0, 0, 0, 0, pad_len))
            C_seq = F.pad(C_seq, (0, 0, 0, 0, 0, pad_len))
            dt_seq = F.pad(dt_seq, (0, 0, 0, pad_len))
        
        x_batch_list.append(x_seq)
        B_batch_list.append(B_seq)
        C_batch_list.append(C_seq)
        dt_batch_list.append(dt_seq)
    
    # Stack to create batch dimension
    x_batch = torch.cat(x_batch_list, dim=0)  # (num_sequences, max_seqlen, nheads, headdim)
    B_batch = torch.cat(B_batch_list, dim=0)  # (num_sequences, max_seqlen, ngroups, dstate)
    C_batch = torch.cat(C_batch_list, dim=0)  # (num_sequences, max_seqlen, ngroups, dstate)
    dt_batch = torch.cat(dt_batch_list, dim=0)  # (num_sequences, max_seqlen, nheads)
    
    print(f"  Batch shapes: x={x_batch.shape}, B={B_batch.shape}, C={C_batch.shape}, dt={dt_batch.shape}")
    
    # Call mamba_chunk_scan_combined once with full batch
    print(f"\n  Processing all sequences in single batch call...")
    out_batch, states_batch = mamba_chunk_scan_combined(
        x_batch, dt_batch, A, B_batch, C_batch, chunk_size,
        D=None, z=None, dt_bias=dt_bias,
        initial_states=initial_states_packed,
        seq_idx=None,
        cu_seqlens=None,
        dt_softplus=True,
        return_final_states=True
    )
    
    print(f"  ✓ Batch processing completed: out_shape={out_batch.shape}, states_shape={states_batch.shape}")
    
    # Unpack results (trim padding and split into individual sequences)
    out_individual_list = []
    states_individual_list = []
    for i in range(num_sequences):
        seq_len = seqlens[i].item()
        out_seq = out_batch[i:i+1, :seq_len, :, :].contiguous()
        state_seq = states_batch[i:i+1, :, :, :].contiguous()
        out_individual_list.append(out_seq)
        states_individual_list.append(state_seq)
    
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

    print(f"out_packed: {out_packed[0][0][1][:10].tolist()}")
    print("--------------------------------")
    print(f"out_individual: {out_individual[0][0][1][:10].tolist()}")
    
    #print("")
    #print(f"out_packed: {states_packed[0][0][0][:10].tolist()}")
    #print("--------------------------------")
    #print(f"states_individual: {states_batch[0][0][0][:10].tolist()}")

    print("Shape of states_packed: ", states_packed.shape)
    print("Shape of states_batch: ", states_batch.shape)
    print("Shape of varlen_states_packed: ", varlen_states_packed.shape)
    print("Shape of states_individual: ", states_individual.shape)

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
