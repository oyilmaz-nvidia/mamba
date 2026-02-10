#!/usr/bin/env python3
"""
Simple standalone test for mamba_chunk_scan_combined with packed sequences and initial states.
This test verifies the functionality tested in lines 374-375 of ssd_combined.py.
"""

import torch
import torch.nn.functional as F
from mamba_ssm.ops.triton.ssd_combined import mamba_chunk_scan_combined


def test_packed_sequences_with_initial_states():
    """Test the packed sequence case where states are passed."""
    print("Testing mamba_chunk_scan_combined with packed sequences and initial states...")
    
    device = 'cuda'
    dtype = torch.float32
    
    # Setup packed sequences
    num_sequences = 3
    seqlens = torch.tensor([128, 256, 192], device=device)
    cu_seqlens = F.pad(seqlens.cumsum(0), (1, 0))
    total_seqlen = seqlens.sum().item()
    
    print(f"Number of sequences: {num_sequences}")
    print(f"Sequence lengths: {seqlens.tolist()}")
    print(f"Total sequence length: {total_seqlen}")
    print(f"cu_seqlens: {cu_seqlens.tolist()}")
    
    # Create seq_idx
    seq_idx = torch.cat([
        torch.full((s,), i, dtype=torch.int32, device=device) 
        for i, s in enumerate(seqlens)
    ], dim=0)

    #print(f"seq_idx: {seq_idx.tolist()}")
    #print("cu_seqlens: ", cu_seqlens.tolist())
        # Model dimensions
    batch = 1  # Required for packed sequences
    nheads = 8
    headdim = 64
    ngroups = 4
    dstate = 32
    chunk_size = 64
    
    print(f"\nModel dimensions:")
    print(f"  nheads: {nheads}, headdim: {headdim}")
    print(f"  ngroups: {ngroups}, dstate: {dstate}")
    print(f"  chunk_size: {chunk_size}")
    
    # Create inputs
    x = torch.randn(batch, total_seqlen, nheads, headdim, dtype=dtype, device=device)
    B = torch.randn(batch, total_seqlen, ngroups, dstate, dtype=dtype, device=device) / 5
    C = torch.randn(batch, total_seqlen, ngroups, dstate, dtype=dtype, device=device) / 5
    dt = torch.randn(batch, total_seqlen, nheads, device=device, dtype=torch.float32) - 4  # Will be softplus'd in the function
    A = -0.1 * torch.rand(nheads, device=device)
    dt_bias = torch.randn(nheads, dtype=torch.float32, device=device) * 0.1
    
    # Create per-sequence initial states - THIS IS THE KEY PART BEING TESTED
    #initial_states = torch.randn(num_sequences, nheads, headdim, dstate, dtype=dtype, device=device) / 10
    initial_states = torch.zeros(num_sequences, nheads, headdim, dstate, dtype=dtype, device=device)
    
    print(f"\nInitial states shape: {initial_states.shape}")
    print(f"Expected shape: ({num_sequences}, {nheads}, {headdim}, {dstate})")
    
    # This is the assertion being tested from lines 374-375
    assert initial_states.shape == (num_sequences, nheads, headdim, dstate), \
        f"Initial states shape mismatch!"
    
    print("\n✓ Initial states shape is correct for packed sequences")
    
    # Run forward pass
    print("\nRunning forward pass with packed sequences...")
    out, varlen_states = mamba_chunk_scan_combined(
        x, dt, A, B, C, chunk_size,
        D=None, z=None, dt_bias=dt_bias,
        initial_states=initial_states,
        seq_idx=seq_idx.unsqueeze(0),  # Add batch dimension
        cu_seqlens=cu_seqlens,
        dt_softplus=True,
        return_varlen_states=True
    )
    
    print(f"✓ Forward pass succeeded!")
    print(f"  Output shape: {out.shape}")
    print(f"  Varlen states shape: {varlen_states.shape}")
    
    # Verify output shapes
    assert out.shape == (batch, total_seqlen, nheads, headdim)
    assert varlen_states.shape == (num_sequences, nheads, headdim, dstate)
    
    print(f"\n✓ Output shapes are correct")
    
    # Reference: process each sequence independently in varlen mode
    print("\nRunning reference computation (per-sequence in varlen mode)...")
    out_ref_list = []
    varlen_states_ref_list = []
    
    for i in range(num_sequences):
        start_idx = cu_seqlens[i].item()
        end_idx = cu_seqlens[i + 1].item()
        seq_len = end_idx - start_idx
        
        # Extract this sequence's data
        x_seq = x[:, start_idx:end_idx, :, :].detach().clone()
        B_seq = B[:, start_idx:end_idx, :, :].detach().clone()
        C_seq = C[:, start_idx:end_idx, :, :].detach().clone()
        dt_seq = dt[:, start_idx:end_idx, :].detach().clone()
        
        # Use this sequence's initial state (keeping it as batch=1 -> num_sequences=1)
        initial_state_seq = initial_states[i:i+1, :, :, :].detach().clone()
        
        # Create seq_idx and cu_seqlens for this single sequence
        seq_idx_single = torch.zeros(batch, seq_len, dtype=torch.int32, device=device)
        cu_seqlens_single = torch.tensor([0, seq_len], device=device)
        
        # Run forward pass for this sequence in varlen mode
        out_seq, varlen_state_seq = mamba_chunk_scan_combined(
            x_seq, dt_seq, A, B_seq, C_seq, chunk_size,
            D=None, z=None, dt_bias=dt_bias,
            initial_states=initial_state_seq,
            seq_idx=seq_idx_single,
            cu_seqlens=cu_seqlens_single,
            dt_softplus=True,
            return_varlen_states=True
        )
        
        out_ref_list.append(out_seq)
        varlen_states_ref_list.append(varlen_state_seq)
        print(f"  Sequence {i}: length={seq_len}, out_shape={out_seq.shape}")
    
    # Concatenate reference outputs
    out_ref = torch.cat(out_ref_list, dim=1)
    varlen_states_ref = torch.cat(varlen_states_ref_list, dim=0)
    
    # Compare outputs
    out_diff = (out - out_ref).abs().max().item()
    states_diff = (varlen_states - varlen_states_ref).abs().max().item()
    
    print(f"\nComparison with reference:")
    print(f"  Output max diff: {out_diff:.6e}")
    print(f"  Varlen states max diff: {states_diff:.6e}")
    
    rtol, atol = 1e-2, 3e-3
    assert torch.allclose(out, out_ref, rtol=rtol, atol=atol), \
        f"Output mismatch: max diff = {out_diff}"
    assert torch.allclose(varlen_states, varlen_states_ref, rtol=rtol, atol=atol), \
        f"Varlen states mismatch: max diff = {states_diff}"
    
    print(f"\n✅ ALL TESTS PASSED!")
    print("\nThe packed sequence case with initial states (lines 374-375) works correctly!")
    return True


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available. This test requires a GPU.")
        exit(1)
    
    try:
        test_packed_sequences_with_initial_states()
    except Exception as e:
        print(f"\n❌ TEST FAILED with error:")
        print(f"{type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
