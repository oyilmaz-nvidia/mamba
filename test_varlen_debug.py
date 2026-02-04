#!/usr/bin/env python3
"""
Debug test to understand the varlen states issue.
"""

import torch
import torch.nn.functional as F
from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined


def test_single_sequence_both_modes():
    """Test a single sequence in both standard and varlen mode to see if they match."""
    print("Testing single sequence in both standard and varlen modes...")
    
    device = 'cuda'
    dtype = torch.float32
    
    # Single sequence
    seqlen = 128
    batch = 1
    nheads = 8
    headdim = 64
    ngroups = 4
    dstate = 32
    chunk_size = 64
    
    # Create inputs
    x = torch.randn(batch, seqlen, nheads, headdim, dtype=dtype, device=device)
    B = torch.randn(batch, seqlen, ngroups, dstate, dtype=dtype, device=device) / 5
    C = torch.randn(batch, seqlen, ngroups, dstate, dtype=dtype, device=device) / 5
    dt = F.softplus(torch.randn(batch, seqlen, nheads, device=device, dtype=torch.float32) - 4)
    A = -0.1 * torch.rand(nheads, device=device)
    dt_bias = torch.randn(nheads, dtype=torch.float32, device=device) * 0.1
    initial_state = torch.randn(batch, nheads, headdim, dstate, dtype=dtype, device=device) / 10
    
    # Standard mode with return_final_states
    print("\n1. Standard mode:")
    out_standard, final_states_standard = mamba_chunk_scan_combined(
        x, dt, A, B, C, chunk_size,
        D=None, z=None, dt_bias=dt_bias,
        initial_states=initial_state,
        seq_idx=None,
        cu_seqlens=None,
        dt_softplus=True,
        return_final_states=True
    )
    print(f"  Output shape: {out_standard.shape}")
    print(f"  Final states shape: {final_states_standard.shape}")
    
    # Varlen mode with return_varlen_states (single sequence)
    print("\n2. Varlen mode (single sequence):")
    cu_seqlens = torch.tensor([0, seqlen], device=device)
    seq_idx = torch.zeros(batch, seqlen, dtype=torch.int32, device=device)
    initial_state_varlen = initial_state  # Shape: (1, nheads, headdim, dstate) - batch becomes num_sequences
    
    print("seq_idx: ", seq_idx.tolist())
    print("cu_seqlens: ", cu_seqlens.tolist())

    out_varlen, varlen_states = mamba_chunk_scan_combined(
        x, dt, A, B, C, chunk_size,
        D=None, z=None, dt_bias=dt_bias,
        initial_states=initial_state_varlen,
        seq_idx=seq_idx,
        cu_seqlens=cu_seqlens,
        dt_softplus=True,
        return_varlen_states=True
    )
    print(f"  Output shape: {out_varlen.shape}")
    print(f"  Varlen states shape: {varlen_states.shape}")
    
    # Compare
    out_diff = (out_standard - out_varlen).abs().max().item()
    states_diff = (final_states_standard - varlen_states).abs().max().item()
    
    print(f"\n3. Comparison:")
    print(f"  Output max diff: {out_diff:.6e}")
    print(f"  States max diff: {states_diff:.6e}")
    
    rtol, atol = 1e-3, 1e-3
    out_match = torch.allclose(out_standard, out_varlen, rtol=rtol, atol=atol)
    states_match = torch.allclose(final_states_standard, varlen_states, rtol=rtol, atol=atol)
    
    print(f"\n  Output match: {out_match}")
    print(f"  States match: {states_match}")
    
    if not out_match or not states_match:
        print("\n❌ MISMATCH! Standard mode and varlen mode produce different results for the same sequence!")
        return False
    else:
        print("\n✅ Standard and varlen modes match for single sequence")
        return True


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available.")
        exit(1)
    
    try:
        test_single_sequence_both_modes()
    except Exception as e:
        print(f"\n❌ TEST FAILED:")
        print(f"{type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
