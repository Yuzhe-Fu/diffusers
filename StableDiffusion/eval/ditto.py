# ditto.py
# Ditto (algorithm-level) for Stable Diffusion UNet/DiT in PyTorch.
# Modes: "fp32" | "fake_dynamic"
# fp32 is full precision; fake_dynamic uses per-channel W8 (from max-abs) and per-tensor A8 (dynamic)
# Features:
#   • INT4 delta execution (split int4 + residual int8) in qdiff mode
#   • Zero-skip emulation (Conv: sparse im2col; Linear: row-skip), with safe fallbacks
#   • Robust per-run cache resets and DEFO layer selection
#   • Attention difference path (float) compatible with recent diffusers

from __future__ import annotations
import time, json
from dataclasses import dataclass
from typing import Any, Dict, Optional, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# Config & Controller
@dataclass
class DittoConfig:
    enable: bool = True

    # Quantization mode
    # Supported: "fp32" | "fake_dynamic"
    #  - fp32: no quantization, baseline execution
    #  - fake_dynamic: W8 per-channel from max-abs; A8 dynamic per-tensor (no calibration JSON)
    quant_mode: str = "fake_dynamic"
    a_bits: int = 8
    w_bits: int = 8
    per_channel_w: bool = True

    # Q-Diff scales are no longer used; kept for compatibility but ignored.
    qd_scales: Optional[Dict[str, Any]] = None  # ignored
    qd_time_bins: int = 1  # ignored for scaling
    total_steps: int = 50  # still used for step tracking

    # Ditto algorithm
    enable_defo: bool = True
    enable_attention_ditto: bool = True
    print_layer_choices: bool = False
    collect_stats: bool = False
    force_diff_all: bool = False
    force_orig_all: bool = False

    # Selection of which modules to quantize/wrap
    # If True, only attention projections (to_q / to_k / to_v / to_out.0) are quantized.
    # If False, all Linear/Conv2d modules are considered.
    quantize_attn_only: bool = False
    # If True, restrict attention quantization to Q-projection (to_q) only.
    quantize_to_q_only: bool = False

    # Mixed-precision delta execution
    delta_int4: bool = True         # INT4 (+ residual int8) delta in qdiff mode
    zero_skip: bool = True          # skip zero-delta work (Conv/Linear)
    sparse_outliers: bool = False   # sparsify residual int8 path too (usually off)

    # Softmax offload (memory-friendly attention). When enabled, compute
    # softmax(scores) on CPU in float32 and stream probability chunks back to GPU
    # to multiply with V. Helps avoid large P tensors on GPU for big sequences.
    softmax_offload: bool = False
    softmax_chunk: int = 2048


_GLOBAL_DITTO: "DittoController" = None
def get_controller() -> "DittoController":
    assert _GLOBAL_DITTO is not None, "Call attach_ditto(pipeline) first."
    return _GLOBAL_DITTO


# Quant helpers
class UniformSymmFakeQuant(nn.Module):
    """
    Symmetric uniform quantizer returning dequantized (q*s).
    If fixed_scale is provided (float/tensor), use it. Otherwise:
      - weights: self.w_scale (calibrated once)
      - activations: dynamic per-tensor scale
    """
    def __init__(self, n_bits: int = 8, is_weight: bool = False,
                 fixed_scale: Optional[torch.Tensor] = None, eps: float = 1e-8):
        super().__init__()
        self.n_bits = n_bits
        self.is_weight = is_weight
        self.eps = eps
        self.register_buffer("w_scale", torch.tensor(1.0))
        self.qmax = (2 ** (n_bits - 1)) - 1
        self.fixed_scale = fixed_scale

    def weight_calibrate(self, w: torch.Tensor):
        if self.fixed_scale is not None:
            return
        with torch.no_grad():
            s = w.abs().max().clamp(min=self.eps) / self.qmax
            self.w_scale.copy_(s)

    def _qd(self, x: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        q = torch.round(x / s).clamp(-self.qmax, self.qmax)
        return q * s

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.fixed_scale is not None:
            s = self.fixed_scale
            if s.ndim == 0:
                return self._qd(x, s)
            shape = [x.shape[0]] + [1]*(x.ndim-1)
            return self._qd(x, s.view(*shape))
        if self.is_weight:
            return self._qd(x, self.w_scale)
        s = x.abs().max().clamp(min=self.eps) / self.qmax
        return self._qd(x, s)

def _quant_codes_int(x: torch.Tensor, scale: torch.Tensor, qmax: int) -> torch.Tensor:
    """Return integer codes (int16) in [-qmax, qmax] using given scalar scale."""
    return torch.clamp(torch.round(x / scale), -qmax, qmax).to(torch.int16)

# Perf / Stats
class LayerPerf:
    def __init__(self):
        self.t1_orig_ms: Optional[float] = None
        self.t2_diff_ms: Optional[float] = None
        self.decision: Optional[str] = None

# Controller
class DittoController:
    def __init__(self, cfg: DittoConfig):
        self.cfg = cfg
        # Enforce fake_dynamic as the only quantization mode (calibration JSON disabled)
        if self.cfg.quant_mode != "fp32":
            if self.cfg.quant_mode != "fake_dynamic":
                print(f"[Ditto] Forcing quant_mode='fake_dynamic' (qd_scales disabled).")
            self.cfg.quant_mode = "fake_dynamic"
        self._step_idx: int = -1
        self._last_scheduler_t: Optional[int] = None
        self._last_batch: Optional[int] = None
        self.layer_perf: Dict[str, LayerPerf] = {}
        self.layer_cache: Dict[str, Dict[str, Any]] = {}

        # Disable Q-Diff scales usage
        self.act_scales: Dict[str, List[float]] = {}
        self.weight_scales: Dict[str, Any] = {}
        self.act_scales_branches: Dict[str, Any] = {}

        # delta_q stats
        self.delta_log: Dict[str, Dict[str, List[float]]] = {}

    def _reset_for_new_run(self):
        self._step_idx = 0
        self.layer_cache.clear()

    def _get_branch_scales(ctrl: DittoController, lid: str, x: torch.Tensor):
        ent = ctrl.act_scales_branches.get(lid)
        if ent is None: return None
        B = max(1, ctrl.cfg.qd_time_bins)
        width = max(1, int((ctrl.cfg.total_steps + B - 1)//B))
        b = min(B - 1, max(0, ctrl.step_idx)//width)
        c1 = int(ent["c1"])
        s1 = torch.tensor(ent["bin_scales_1"][b], dtype=x.dtype, device=x.device).clamp(min=1e-8)
        s2 = torch.tensor(ent["bin_scales_2"][b], dtype=x.dtype, device=x.device).clamp(min=1e-8)
        return c1, s1, s2

    def on_unet_call(self, timestep_tensor: torch.Tensor, sample: Optional[torch.Tensor] = None):
        t_val = int(timestep_tensor.flatten()[0].item()) if isinstance(timestep_tensor, torch.Tensor) else int(timestep_tensor)
        bsz = int(sample.shape[0]) if (isinstance(sample, torch.Tensor) and sample.ndim >= 1) else None

        new_run = False
        if self._last_scheduler_t is None:
            new_run = True
        else:
            if t_val > self._last_scheduler_t:
                new_run = True
            if (bsz is not None) and (self._last_batch is not None) and (bsz != self._last_batch):
                new_run = True

        if new_run:
            self._reset_for_new_run()
        else:
            if t_val != self._last_scheduler_t:
                self._step_idx += 1

        self._last_scheduler_t = t_val
        if bsz is not None:
            self._last_batch = bsz

    @property
    def step_idx(self) -> int:
        return self._step_idx

    def cache_of(self, layer_id: str) -> Dict[str, Any]:
        if layer_id not in self.layer_cache:
            self.layer_cache[layer_id] = {}
        return self.layer_cache[layer_id]

    def perf_of(self, layer_id: str) -> LayerPerf:
        if layer_id not in self.layer_perf:
            self.layer_perf[layer_id] = LayerPerf()
        return self.layer_perf[layer_id]

    def layer_should_use_diff(self, layer_id: str) -> bool:
        if self.cfg.force_diff_all: return self.step_idx >= 1
        if self.cfg.force_orig_all: return False
        perf = self.perf_of(layer_id)
        if perf.decision is not None:
            return perf.decision == "diff"
        if self.step_idx == 0:
            return False
        if self.step_idx == 1:
            return True
        return True

    def record_time(self, layer_id: str, mode: str, ms: float):
        perf = self.perf_of(layer_id)
        if self.step_idx == 0 and mode == "orig":
            perf.t1_orig_ms = ms
        elif self.step_idx == 1 and mode == "diff":
            perf.t2_diff_ms = ms
            if self.cfg.enable_defo and perf.t1_orig_ms is not None:
                perf.decision = "diff" if perf.t2_diff_ms <= perf.t1_orig_ms else "orig"
                if self.cfg.print_layer_choices:
                    print(f"[DEFO] {layer_id}: t1={perf.t1_orig_ms:.2f}ms, t2={perf.t2_diff_ms:.2f}ms -> {perf.decision}")

    # delta_q stats
    def add_delta_stats(self, layer_id: str, zeros_frac: float, within4_frac: float, numel: int):
        d = self.delta_log.setdefault(layer_id, {"zeros": [], "within4": [], "N": []})
        d["zeros"].append(float(zeros_frac)); d["within4"].append(float(within4_frac)); d["N"].append(int(numel))

    def dump_delta_stats(self, save_path: str):
        out = {}
        for k, d in self.delta_log.items():
            Ntot = sum(d["N"]) if d["N"] else 0
            z_mean = (sum(z*n for z, n in zip(d["zeros"], d["N"])) / Ntot) if Ntot else 0.0
            w4_mean = (sum(w*n for w, n in zip(d["within4"], d["N"])) / Ntot) if Ntot else 0.0
            out[k] = {
                "steps": len(d["N"]),
                "elements_total": Ntot,
                "zero_ratio_weighted": z_mean,
                "within4_ratio_weighted": w4_mean,
            }
        with open(save_path, "w") as f:
            json.dump(out, f, indent=2)
        print(f"[Ditto] Δq stats saved -> {save_path}")

    def attach(self, pipeline: Any):
        global _GLOBAL_DITTO
        _GLOBAL_DITTO = self
        # Choose core denoiser module: UNet (SD1.x/XL) or Transformer (SD3.x)
        if hasattr(pipeline, "unet") and isinstance(getattr(pipeline, "unet"), nn.Module):
            core = pipeline.unet
            mode = "unet"
        elif hasattr(pipeline, "transformer") and isinstance(getattr(pipeline, "transformer"), nn.Module):
            core = pipeline.transformer
            mode = "transformer"
        else:
            raise AttributeError("Ditto attach: pipeline has neither .unet nor .transformer")

        if not hasattr(core, "_ditto_orig_forward"):
            core._ditto_orig_forward = core.forward

            if mode == "unet":
                def _wrap(sample, timestep, **kwargs):
                    self.on_unet_call(timestep, sample)
                    return core._ditto_orig_forward(sample, timestep, **kwargs)
            else:
                def _wrap(hidden_states, encoder_hidden_states=None, pooled_projections=None, timestep=None, **kwargs):
                    self.on_unet_call(timestep, hidden_states)
                    return core._ditto_orig_forward(
                        hidden_states,
                        encoder_hidden_states=encoder_hidden_states,
                        pooled_projections=pooled_projections,
                        timestep=timestep,
                        **kwargs,
                    )

            core.forward = _wrap

    def detach(self, pipeline: Any):
        global _GLOBAL_DITTO
        _GLOBAL_DITTO = None
        core = None
        if hasattr(pipeline, "unet") and hasattr(pipeline.unet, "_ditto_orig_forward"):
            core = pipeline.unet
        elif hasattr(pipeline, "transformer") and hasattr(pipeline.transformer, "_ditto_orig_forward"):
            core = pipeline.transformer
        if core is not None:
            core.forward = core._ditto_orig_forward
            delattr(core, "_ditto_orig_forward")


# Traversal and scale getters
def _named_modules_with_types(m: nn.Module, types, prefix=""):
    for name, child in m.named_children():
        full = f"{prefix}.{name}" if prefix else name
        if isinstance(child, types):
            yield full, child
        yield from _named_modules_with_types(child, types, prefix=full)

def _get_fixed_weight_scale(ctrl: DittoController, lid: str, w: torch.Tensor) -> Optional[torch.Tensor]:
    if ctrl.cfg.quant_mode != "qdiff": return None
    entry = ctrl.weight_scales.get(lid, None)
    if entry is None: return None
    if isinstance(entry, list):
        return torch.tensor(entry, dtype=w.dtype, device=w.device).clamp(min=1e-8)
    return torch.tensor(entry, dtype=w.dtype, device=w.device).clamp(min=1e-8)

def _get_fixed_act_scale(ctrl: DittoController, lid: str, x: torch.Tensor) -> Optional[torch.Tensor]:
    if ctrl.cfg.quant_mode != "qdiff": return None
    scales = ctrl.act_scales.get(lid, None)
    if scales is None: return None
    B = max(1, ctrl.cfg.qd_time_bins)
    if B <= 1:
        # Support both scalar and list formats
        if isinstance(scales, (list, tuple)):
            val = scales[0] if len(scales) > 0 else 1.0
        else:
            val = float(scales)
        s = torch.tensor(val, dtype=x.dtype, device=x.device)
    else:
        # time-binned scales must be a list
        if not isinstance(scales, (list, tuple)):
            val = float(scales)
            s = torch.tensor(val, dtype=x.dtype, device=x.device)
        else:
            width = max(1, int((ctrl.cfg.total_steps + B - 1) // B))
            b = min(B - 1, max(0, ctrl.step_idx) // width)
            idx = min(max(0, b), len(scales) - 1)
            s = torch.tensor(scales[idx], dtype=x.dtype, device=x.device)
    return s.clamp(min=1e-8)


# Sparse helpers (commented out to disable skipping paths)
def _to_pair(v):
    if isinstance(v, tuple): return v
    return (v, v)

def _conv2d_sparse_im2col(delta: torch.Tensor, weight: torch.Tensor,
                          stride: Tuple[int,int], padding: Tuple[int,int], dilation: Tuple[int,int]) -> torch.Tensor:
    """
    Sparse "delta conv" using im2col selection. Only columns with any non-zero are computed.
    Limitation: groups==1 and dilation==(1,1). Fallback to dense elsewhere.
    """
    B, Cin, H, W = delta.shape
    Cout, _, kH, kW = weight.shape
    if dilation != (1,1):
        return F.conv2d(delta, weight, None, stride, padding, dilation, 1)

    # Unfold per batch to columns
    x_unf = F.unfold(delta, kernel_size=(kH,kW), padding=padding, stride=stride)  # [B, Cin*kH*kW, L]
    B_, K, L = x_unf.shape
    out_h = (H + 2*padding[0] - kH) // stride[0] + 1
    out_w = (W + 2*padding[1] - kW) // stride[1] + 1
    W_flat = weight.view(Cout, -1)  # [Cout, K]

    y_unf = delta.new_zeros((B, Cout, L), dtype=delta.dtype)
    for b in range(B):
        cols = x_unf[b]                           # [K, L]
        nz_mask = (cols != 0).any(dim=0)          # [L]
        if nz_mask.any():
            idx = nz_mask.nonzero(as_tuple=False).squeeze(1)  # [Lnz]
            y_sel = (W_flat @ cols[:, idx].to(weight.dtype))  # [Cout, Lnz]
            y_unf[b, :, idx] = y_sel.to(delta.dtype)

    # Fold back (each column maps to single output location)
    return y_unf.view(B, Cout, out_h, out_w)

def _linear_delta_sparse_anyrank(delta: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """
    Sparse linear with row-skip for inputs of shape (..., In). Returns (..., Out).
    """
    *lead, In = delta.shape
    Out = w.shape[0]
    if In != w.shape[1]:
        raise RuntimeError(f"[DittoLinear] shape mismatch: delta[...,{In}] vs weight[{w.shape}]")
    x2 = delta.reshape(-1, In)            # [Bflat, In]
    nz_rows = (x2 != 0).any(dim=1)        # [Bflat]
    out2 = delta.new_zeros((x2.shape[0], Out), dtype=delta.dtype)
    if nz_rows.any():
        out2[nz_rows] = F.linear(x2[nz_rows], w, None)
    return out2.reshape(*lead, Out)



# Ditto for Convolution layer
class DittoConv2d(nn.Module):
    def __init__(self, conv: nn.Conv2d, layer_id: str, cfg: DittoConfig):
        super().__init__()
        self.conv = conv
        self.id = layer_id
        self.cfg = cfg

        # Activation quantizer: present for qdiff and fake_dynamic
        self.aq = UniformSymmFakeQuant(cfg.a_bits, False) if cfg.quant_mode in ("qdiff", "fake_dynamic") else None

        # Weight centers (W8)
        fixed_w = None
        if cfg.quant_mode == "qdiff" and cfg.qd_scales is not None:
            entry = cfg.qd_scales.get("weight_scales", {}).get(self.id, None)
            if entry is not None:
                fixed_w = torch.tensor(entry, dtype=conv.weight.dtype, device=conv.weight.device)
        # For fake_dynamic, create a quantizer without fixed scale and calibrate from weights
        if cfg.quant_mode in ("qdiff", "fake_dynamic"):
            self.wq = UniformSymmFakeQuant(cfg.w_bits, True, fixed_scale=fixed_w)
        else:
            self.wq = None
        if self.wq is not None and self.wq.fixed_scale is None:
            self.wq.weight_calibrate(conv.weight.data)

    def _act_scale(self, x: torch.Tensor) -> Optional[torch.Tensor]:
        ctrl = get_controller()
        if ctrl.cfg.quant_mode == "qdiff":
            s = _get_fixed_act_scale(ctrl, self.id, x)
            if s is not None: return s
        if self.aq is None: return None
        with torch.no_grad():
            qmax = (2 ** (self.cfg.a_bits - 1)) - 1
            return x.detach().abs().max().clamp(min=1e-8) / qmax

    def _quant_act(self, x: torch.Tensor, s: Optional[torch.Tensor]) -> torch.Tensor:
        if self.aq is None: return x
        if s is not None:
            return UniformSymmFakeQuant(self.cfg.a_bits, False, fixed_scale=s)(x)
        return self.aq(x)

    def _quant_w(self) -> torch.Tensor:
        if self.wq is None: return self.conv.weight
        if self.wq.fixed_scale is not None and self.wq.fixed_scale.ndim == 1:
            s = self.wq.fixed_scale.view(-1, *([1]*(self.conv.weight.ndim-1)))
            q = torch.round(self.conv.weight / s).clamp(-self.wq.qmax, self.wq.qmax)
            return q * s
        return self.wq(self.conv.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Ditto Conv2d delta execution with Q-Diffusion scales.
        - Step 0: A8/W8 (dequantized) dense conv, cache prev_x/prev_y.
        - Step ≥1: integer delta_q on current A8 grid, INT4 <= int4 + int8 residual, zero-skip optional.
        - If concat-split branch scales exist, quantize/dequantize each branch separately.
        """
        ctrl = get_controller()
        cache = ctrl.cache_of(self.id)

        w = self._quant_w() 
        ent = getattr(ctrl, "act_scales_branches", {}).get(self.id, None)
        has_branch = (ent is not None) and (x.dim() == 4) and (0 < int(ent.get("c1", 0)) < x.shape[1])

        if has_branch:
            # Per-branch A8 scales (time-binned)
            B = max(1, ctrl.cfg.qd_time_bins)
            width = max(1, int((ctrl.cfg.total_steps + B - 1) // B))
            bidx = min(B - 1, max(0, ctrl.step_idx) // width)
            c1 = int(ent["c1"])
            s1 = torch.tensor(ent["bin_scales_1"][bidx], dtype=x.dtype, device=x.device).clamp(min=1e-8)
            s2 = torch.tensor(ent["bin_scales_2"][bidx], dtype=x.dtype, device=x.device).clamp(min=1e-8)
            # Step-0 quantized/dequantized activation (per branch)
            x1, x2 = x[:, :c1], x[:, c1:]
            q1 = UniformSymmFakeQuant(self.cfg.a_bits, False, fixed_scale=s1)(x1)
            q2 = UniformSymmFakeQuant(self.cfg.a_bits, False, fixed_scale=s2)(x2)
            qact = torch.cat([q1, q2], dim=1)
            single_scale = None  
        else:
            # Single A8 scale (time-binned or scalar)
            s_a = _get_fixed_act_scale(ctrl, self.id, x)
            if s_a is None and self.aq is not None:
                # dynamic fallback (should not trigger in qdiff when scales exist)
                with torch.no_grad():
                    qmax = (2 ** (self.cfg.a_bits - 1)) - 1
                    s_a = x.detach().abs().max().clamp(min=1e-8) / qmax
            qact = self._quant_act(x, s_a) if self.aq is not None else x
            single_scale = s_a  # scalar or None

        # Step-0: fupp precision (fp16) path
        if (ctrl.step_idx == 0) or (not ctrl.layer_should_use_diff(self.id)) or (ctrl.cfg.quant_mode == "fp32"):
            t0 = time.perf_counter()
            y = F.conv2d(qact, w, self.conv.bias, self.conv.stride, self.conv.padding,
                        self.conv.dilation, self.conv.groups)
            if x.is_cuda: torch.cuda.synchronize()
            ctrl.record_time(self.id, "orig", (time.perf_counter() - t0) * 1000)
            cache["prev_x"], cache["prev_y"] = qact.detach(), y.detach()
            if single_scale is not None:
                cache["prev_scale"] = single_scale.detach().clone()
            return y

        # Step 1 and later: delta path 
        prev_x = cache.get("prev_x", None)
        prev_y = cache.get("prev_y", None)
        if (prev_x is None) or (prev_y is None) or (prev_x.shape[0] != qact.shape[0]):
            # Fallback to orig if caches are invalid for any reason
            t0 = time.perf_counter()
            y = F.conv2d(qact, w, self.conv.bias, self.conv.stride, self.conv.padding,
                        self.conv.dilation, self.conv.groups)
            if x.is_cuda: torch.cuda.synchronize()
            ctrl.record_time(self.id, "orig", (time.perf_counter() - t0) * 1000)
            cache["prev_x"], cache["prev_y"] = qact.detach(), y.detach()
            return y

        qmax8 = (2 ** (self.cfg.a_bits - 1)) - 1
        if has_branch:
            x1, x2 = x[:, :c1], x[:, c1:]
            px1, px2 = prev_x[:, :c1], prev_x[:, c1:]
            q_t   = torch.cat([_quant_codes_int(x1,  s1, qmax8),
                            _quant_codes_int(x2,  s2, qmax8)], dim=1)
            q_tm1 = torch.cat([_quant_codes_int(px1, s1, qmax8),
                            _quant_codes_int(px2, s2, qmax8)], dim=1)
            dq = q_t - q_tm1  # int16
        else:
            s_a = single_scale
            if s_a is None:
                with torch.no_grad():
                    qmax = (2 ** (self.cfg.a_bits - 1)) - 1
                    # In fake_dynamic, prefer current float x range for scale
                    s_a = x.detach().abs().max().clamp(min=1e-8) / qmax
            q_t   = _quant_codes_int(x,     s_a, qmax8)
            q_tm1 = _quant_codes_int(prev_x, s_a, qmax8)
            dq = q_t - q_tm1  # int16

        # Stats from integer delta (exact zeros, exact inliers)
        if ctrl.cfg.collect_stats:
            di = dq.detach()
            zeros = (di == 0).float().mean().item()
            within4 = ((di >= -8) & (di <= 7)).float().mean().item()
            ctrl.add_delta_stats(self.id, zeros, within4, int(di.numel()))

        # INT4 split (masked inliers; residual in 8-bit)
        if ctrl.cfg.delta_int4:
            out_mask = (dq < -8) | (dq > 7)
            dq4 = torch.where(out_mask, torch.zeros_like(dq), torch.clamp(dq, -8, 7))
            dq8 = torch.where(out_mask, dq, torch.zeros_like(dq))
        else:
            dq4 = dq
            dq8 = dq.new_zeros(())

        # Dequantize deltas back to float (per-branch or single scale)
        if has_branch:
            dq4_1, dq4_2 = dq4[:, :c1], dq4[:, c1:]
            dq8_1, dq8_2 = dq8[:, :c1], dq8[:, c1:]
            dx4 = torch.cat([dq4_1.to(x.dtype) * s1, dq4_2.to(x.dtype) * s2], dim=1)
            dx8 = torch.cat([dq8_1.to(x.dtype) * s1, dq8_2.to(x.dtype) * s2], dim=1)
        else:
            dx4 = dq4.to(x.dtype) * s_a
            dx8 = dq8.to(x.dtype) * s_a

        # Delta conv(s): zero-skip emulation where feasible
        t0 = time.perf_counter()
        if ctrl.cfg.zero_skip and self.conv.groups == 1 and self.conv.dilation == (1, 1):
            y_delta4 = _conv2d_sparse_im2col(dx4, w, _to_pair(self.conv.stride),
                                            _to_pair(self.conv.padding), _to_pair(self.conv.dilation))
        else:
            y_delta4 = F.conv2d(dx4, w, None, self.conv.stride, self.conv.padding, self.conv.dilation, self.conv.groups)

        if ctrl.cfg.sparse_outliers and self.conv.groups == 1 and self.conv.dilation == (1, 1):
            y_delta8 = _conv2d_sparse_im2col(dx8, w, _to_pair(self.conv.stride),
                                            _to_pair(self.conv.padding), _to_pair(self.conv.dilation))
        else:
            y_delta8 = F.conv2d(dx8, w, None, self.conv.stride, self.conv.padding, self.conv.dilation, self.conv.groups)

        y = prev_y + y_delta4 + y_delta8
        if x.is_cuda: torch.cuda.synchronize()
        ctrl.record_time(self.id, "diff", (time.perf_counter() - t0) * 1000)

        # Update caches for next step
        cache["prev_x"], cache["prev_y"] = qact.detach(), y.detach()
        if not has_branch and single_scale is not None:
            cache["prev_scale"] = single_scale.detach().clone()
        return y



# Ditto for Linear layers
class DittoLinear(nn.Module):
    def __init__(self, lin: nn.Linear, layer_id: str, cfg: DittoConfig):
        super().__init__()
        self.lin = lin
        self.id = layer_id
        self.cfg = cfg

        self.aq = UniformSymmFakeQuant(cfg.a_bits, False) if cfg.quant_mode in ("qdiff", "fake_dynamic") else None
        fixed_w = None
        if cfg.quant_mode == "qdiff" and cfg.qd_scales is not None:
            entry = cfg.qd_scales.get("weight_scales", {}).get(self.id, None)
            if entry is not None:
                fixed_w = torch.tensor(entry, dtype=lin.weight.dtype, device=lin.weight.device)
        if cfg.quant_mode in ("qdiff", "fake_dynamic"):
            self.wq = UniformSymmFakeQuant(cfg.w_bits, True, fixed_scale=fixed_w)
        else:
            self.wq = None
        if self.wq is not None and self.wq.fixed_scale is None:
            self.wq.weight_calibrate(lin.weight.data)

    def _act_scale(self, x: torch.Tensor) -> Optional[torch.Tensor]:
        ctrl = get_controller()
        if ctrl.cfg.quant_mode == "qdiff":
            s = _get_fixed_act_scale(ctrl, self.id, x)
            if s is not None: return s
        if self.aq is None: return None
        with torch.no_grad():
            qmax = (2 ** (self.cfg.a_bits - 1)) - 1
            return x.detach().abs().max().clamp(min=1e-8) / qmax

    def _quant_act(self, x: torch.Tensor, s: Optional[torch.Tensor]) -> torch.Tensor:
        if self.aq is None: return x
        if s is not None:
            return UniformSymmFakeQuant(self.cfg.a_bits, False, fixed_scale=s)(x)
        return self.aq(x)

    def _quant_w(self) -> torch.Tensor:
        if self.wq is None: return self.lin.weight
        if self.wq.fixed_scale is not None and self.wq.fixed_scale.ndim == 1:
            s = self.wq.fixed_scale.view(-1, 1)
            q = torch.round(self.lin.weight / s).clamp(-self.wq.qmax, self.wq.qmax)
            return q * s
        return self.wq(self.lin.weight)

    def _linear_delta_sparse(self, delta: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        return F.linear(delta, w, None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ctrl = get_controller()
        cache = ctrl.cache_of(self.id)

        s_a = self._act_scale(x)
        qact = self._quant_act(x, s_a)
        w = self._quant_w()

        def _orig(q):
            t0 = time.perf_counter()
            y_ = F.linear(q, w, self.lin.bias)
            if q.is_cuda: torch.cuda.synchronize()
            ctrl.record_time(self.id, "orig", (time.perf_counter() - t0) * 1000)
            cache["prev_x"], cache["prev_y"] = q.detach(), y_.detach()
            if s_a is not None: cache["prev_scale"] = s_a.detach().clone()
            return y_

        if (ctrl.step_idx == 0) or (not ctrl.layer_should_use_diff(self.id)) or (ctrl.cfg.quant_mode == "fp32"):
            return _orig(qact)

        prev_x, prev_y = cache.get("prev_x"), cache.get("prev_y")
        if (prev_x is None) or (prev_y is None) or (prev_x.shape[0] != qact.shape[0]):
            return _orig(qact)

        if s_a is None:
            with torch.no_grad():
                qmax = (2 ** (self.cfg.a_bits - 1)) - 1
                # In fake_dynamic, prefer current float x range for scale
                s_a = x.detach().abs().max().clamp(min=1e-8) / qmax

        qmax8 = (2 ** (self.cfg.a_bits - 1)) - 1
        q_t   = _quant_codes_int(x,     s_a, qmax8)
        q_tm1 = _quant_codes_int(prev_x, s_a, qmax8)
        dq    = q_t - q_tm1

        if ctrl.cfg.collect_stats:
            di = dq.detach()
            zeros = (di == 0).float().mean().item()
            within4 = ((di >= -8) & (di <= 7)).float().mean().item()
            ctrl.add_delta_stats(self.id, zeros, within4, int(di.numel()))

        if ctrl.cfg.delta_int4:
            out_mask = (dq < -8) | (dq > 7)
            dq4 = torch.where(out_mask, torch.zeros_like(dq), torch.clamp(dq, -8, 7))
            dq8 = torch.where(out_mask, dq, torch.zeros_like(dq))
        else:
            dq4 = dq
            dq8 = dq.new_zeros(())

        dx4 = dq4.to(x.dtype) * s_a
        dx8 = dq8.to(x.dtype) * s_a

        t0 = time.perf_counter()
        if ctrl.cfg.zero_skip:
            y_delta4 = self._linear_delta_sparse(dx4, w)
        else:
            y_delta4 = F.linear(dx4, w, None)

        if ctrl.cfg.sparse_outliers and (dq8.numel() > 0):
            y_delta8 = self._linear_delta_sparse(dx8, w)
        else:
            y_delta8 = F.linear(dx8, w, None)

        y = prev_y + y_delta4 + y_delta8
        if x.is_cuda: torch.cuda.synchronize()
        ctrl.record_time(self.id, "diff", (time.perf_counter() - t0) * 1000)

        cache["prev_x"], cache["prev_y"] = qact.detach(), y.detach()
        if s_a is not None: cache["prev_scale"] = s_a.detach().clone()
        return y


# --- PATCH: quantization-only Linear for attn projections (no Δ) ---

class DittoLinearQuantOnly(nn.Module):
    def __init__(self, lin: nn.Linear, layer_id: str, cfg: DittoConfig):
        super().__init__()
        self.lin = lin
        self.id = layer_id
        self.cfg = cfg

        self.aq = UniformSymmFakeQuant(cfg.a_bits, False) if cfg.quant_mode in ("qdiff", "fake_dynamic") else None
        fixed_w = None
        if cfg.quant_mode == "qdiff" and cfg.qd_scales is not None:
            entry = cfg.qd_scales.get("weight_scales", {}).get(self.id, None)
            if entry is not None:
                fixed_w = torch.tensor(entry, dtype=lin.weight.dtype, device=lin.weight.device)
        self.wq = UniformSymmFakeQuant(cfg.w_bits, True, fixed_scale=fixed_w) if cfg.quant_mode in ("qdiff", "fake_dynamic") else None
        if self.wq is not None and self.wq.fixed_scale is None:
            self.wq.weight_calibrate(lin.weight.data)

    def _act_scale(self, x: torch.Tensor) -> Optional[torch.Tensor]:
        ctrl = get_controller()
        if ctrl.cfg.quant_mode == "qdiff":
            s = _get_fixed_act_scale(ctrl, self.id, x)
            if s is not None: return s
        if self.aq is None: return None
        with torch.no_grad():
            qmax = (2 ** (self.cfg.a_bits - 1)) - 1
            return x.detach().abs().max().clamp(min=1e-8) / qmax

    def _quant_act(self, x: torch.Tensor, s: Optional[torch.Tensor]) -> torch.Tensor:
        if self.aq is None: return x
        if s is not None:
            return UniformSymmFakeQuant(self.cfg.a_bits, False, fixed_scale=s)(x)
        return self.aq(x)

    def _quant_w(self) -> torch.Tensor:
        if self.wq is None: return self.lin.weight
        if self.wq.fixed_scale is not None and self.wq.fixed_scale.ndim == 1:
            s = self.wq.fixed_scale.view(-1, 1)
            q = torch.round(self.lin.weight / s).clamp(-self.wq.qmax, self.wq.qmax)
            return q * s
        return self.wq(self.lin.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s_a = self._act_scale(x)
        qact = self._quant_act(x, s_a)
        w = self._quant_w()
        return F.linear(qact, w, self.lin.bias)


# Attention processor (float diff path)
# Helper: CPU softmax offload with streaming P@V back to GPU
def _softmax_cpu_stream_matmul(scores: torch.Tensor, v: torch.Tensor, chunk: int) -> torch.Tensor:
    """
    Compute softmax(scores) on CPU (float32) and stream probability chunks
    back to GPU to multiply with V on device.
    scores: [B, H, Nq, Nk] on GPU; v: [B, H, Nk, dh] on GPU.
    Returns [B, H, Nq, dh] on GPU.
    """
    scores_cpu = scores.detach().float().cpu()
    p_cpu = F.softmax(scores_cpu, dim=-1)
    del scores_cpu
    B, H, Nq, Nk = p_cpu.shape
    dh = v.shape[-1]
    out = torch.zeros((B, H, Nq, dh), device=v.device, dtype=v.dtype)
    step = max(1, int(chunk))
    for j in range(0, Nk, step):
        j2 = min(Nk, j + step)
        p_chunk = p_cpu[:, :, :, j:j2].to(device=v.device, dtype=v.dtype, non_blocking=True)
        v_chunk = v[:, :, j:j2, :]
        out += torch.matmul(p_chunk, v_chunk)
        del p_chunk
    del p_cpu
    return out
try:
    from diffusers.models.attention_processor import AttnProcessor2_0 as _BaseAttnProc  # type: ignore
    from diffusers.models.attention_processor import JointAttnProcessor2_0 as _BaseJointProc  # type: ignore
    try:
        from diffusers.models.attention_processor import FusedJointAttnProcessor2_0 as _BaseFusedJointProc  # type: ignore
    except Exception:
        _BaseFusedJointProc = None
    DIFFUSERS_AVAILABLE = True
except Exception:
    try:
        import diffusers  # noqa: F401
        DIFFUSERS_AVAILABLE = True
    except Exception:
        DIFFUSERS_AVAILABLE = False
    _BaseAttnProc = nn.Module
    _BaseJointProc = nn.Module
    _BaseFusedJointProc = None

class DittoAttentionProcessor(_BaseAttnProc):
    def __init__(self, layer_id: str, cfg: DittoConfig):
        super().__init__()
        self.id = layer_id
        self.cfg = cfg

    def __call__(self, attn, hidden_states, encoder_hidden_states=None,
                 attention_mask=None, temb=None, **kwargs):
        ctrl = get_controller()
        cache = ctrl.cache_of(self.id)

        q = attn.to_q(hidden_states)
        is_cross = encoder_hidden_states is not None
        k = attn.to_k(encoder_hidden_states if is_cross else hidden_states)
        v = attn.to_v(encoder_hidden_states if is_cross else hidden_states)

        b, n, d = q.shape
        h = attn.heads
        dh = d // h
        q = q.view(b, n, h, dh).transpose(1, 2)
        k = k.view(b, -1, h, dh).transpose(1, 2)
        v = v.view(b, -1, h, dh).transpose(1, 2)
        scale = getattr(attn, "scale", dh ** -0.5)

        def _orig():
            scores_raw = torch.matmul(q, k.transpose(-2, -1)) * scale
            scores = scores_raw
            if attention_mask is not None:
                scores = scores + attention_mask
            if getattr(self.cfg, "softmax_offload", False):
                out4 = _softmax_cpu_stream_matmul(scores, v, getattr(self.cfg, "softmax_chunk", 2048))
                out = out4.transpose(1, 2).reshape(b, n, d)
            else:
                p = scores.float().softmax(dim=-1).to(v.dtype)
                out = torch.matmul(p, v).transpose(1, 2).reshape(b, n, d)
            out = attn.to_out[0](out)
            out = attn.to_out[1](out) if len(attn.to_out) > 1 else out
            # Avoid caching gigantic prev_scores when delta path is disabled (baseline W8A8)
            if not self.cfg.force_orig_all:
                cache["prev_q"], cache["prev_k"], cache["prev_scores"] = q.detach(), k.detach(), scores_raw.detach()
            return out

        use_diff = self.cfg.enable_attention_ditto and (ctrl.step_idx >= 1) and not self.cfg.force_orig_all
        if not use_diff:
            return _orig()

        prev_q, prev_k, prev_scores = cache.get("prev_q"), cache.get("prev_k"), cache.get("prev_scores")
        if (prev_q is None) or (prev_k is None) or (prev_scores is None) or (prev_q.shape != q.shape) or (prev_k.shape != k.shape):
            return _orig()

        # Two-term Δ update on RAW scores, with optional INT4/INT8 delta quantization
        if (ctrl.cfg.quant_mode in ("qdiff", "fake_dynamic")) and self.cfg.delta_int4:
            qmax8 = (2 ** (self.cfg.a_bits - 1)) - 1
            with torch.no_grad():
                s_q = q.detach().abs().max().clamp(min=1e-8) / qmax8
            dq_codes = torch.clamp(torch.round((q - prev_q) / s_q), -qmax8, qmax8).to(torch.int16)
            out_q = (dq_codes < -8) | (dq_codes > 7)
            dq4 = torch.where(out_q, torch.zeros_like(dq_codes), torch.clamp(dq_codes, -8, 7)).to(q.dtype) * s_q
            dq8 = torch.where(out_q, dq_codes, torch.zeros_like(dq_codes)).to(q.dtype) * s_q

            if is_cross:
                scores_raw = prev_scores + (torch.matmul(dq4, k.transpose(-2, -1)) +
                                            torch.matmul(dq8, k.transpose(-2, -1))) * scale
            else:
                qmax8 = (2 ** (self.cfg.a_bits - 1)) - 1
                with torch.no_grad():
                    s_k = k.detach().abs().max().clamp(min=1e-8) / qmax8
                dk_codes = torch.clamp(torch.round((k - prev_k) / s_k), -qmax8, qmax8).to(torch.int16)
                out_k = (dk_codes < -8) | (dk_codes > 7)
                dk4 = torch.where(out_k, torch.zeros_like(dk_codes), torch.clamp(dk_codes, -8, 7)).to(k.dtype) * s_k
                dk8 = torch.where(out_k, dk_codes, torch.zeros_like(dk_codes)).to(k.dtype) * s_k

                scores_raw = prev_scores + (
                    torch.matmul(prev_q, dk4.transpose(-2, -1)) +
                    torch.matmul(prev_q, dk8.transpose(-2, -1)) +
                    torch.matmul(dq4,    k.transpose(-2, -1)) +
                    torch.matmul(dq8,    k.transpose(-2, -1))
                ) * scale
        else:
            dq = q - prev_q
            if is_cross:
                scores_raw = prev_scores + torch.matmul(dq, k.transpose(-2, -1)) * scale
            else:
                dk = k - prev_k
                scores_raw = prev_scores + (torch.matmul(prev_q, dk.transpose(-2, -1)) +
                                            torch.matmul(dq,     k.transpose(-2, -1))) * scale

        scores = scores_raw
        if attention_mask is not None:
            scores = scores + attention_mask
        if getattr(self.cfg, "softmax_offload", False):
            out4 = _softmax_cpu_stream_matmul(scores, v, getattr(self.cfg, "softmax_chunk", 2048))
            out = out4.transpose(1, 2).reshape(b, n, d)
        else:
            p = scores.float().softmax(dim=-1).to(v.dtype)
            out = torch.matmul(p, v).transpose(1, 2).reshape(b, n, d)
        out = attn.to_out[0](out)
        out = attn.to_out[1](out) if len(attn.to_out) > 1 else out

        # Recompute and cache RAW scores for the next step
        new_scores_raw = torch.matmul(q, k.transpose(-2, -1)) * scale
        cache["prev_q"], cache["prev_k"], cache["prev_scores"] = q.detach(), k.detach(), new_scores_raw.detach()
        return out

class DittoJointAttentionProcessor(_BaseJointProc):
    def __init__(self, layer_id: str, cfg: DittoConfig):
        super().__init__()
        self.id = layer_id
        self.cfg = cfg

    def __call__(self, attn, hidden_states, encoder_hidden_states=None,
                 attention_mask=None, temb=None, **kwargs):
        ctrl = get_controller()
        cache = ctrl.cache_of(self.id)
        residual = hidden_states

        # Single-stream (self-attention) branch
        if encoder_hidden_states is None:
            bsz = hidden_states.shape[0]
            q = attn.to_q(hidden_states)
            k = attn.to_k(hidden_states)
            v = attn.to_v(hidden_states)
            inner_dim = k.shape[-1]
            head_dim = inner_dim // attn.heads
            q = q.view(bsz, -1, attn.heads, head_dim).transpose(1, 2)
            k = k.view(bsz, -1, attn.heads, head_dim).transpose(1, 2)
            v = v.view(bsz, -1, attn.heads, head_dim).transpose(1, 2)
            scale = head_dim ** -0.5

            def _orig_single():
                scores_raw = torch.matmul(q, k.transpose(-2, -1)) * scale
                scores = scores_raw
                if attention_mask is not None:
                    scores = scores + attention_mask
                if getattr(self.cfg, "softmax_offload", False):
                    out4 = _softmax_cpu_stream_matmul(scores, v, getattr(self.cfg, "softmax_chunk", 2048))
                    out = out4.transpose(1, 2).reshape(bsz, -1, attn.heads * head_dim)
                    out = out.to(q.dtype)
                else:
                    p = scores.float().softmax(dim=-1).to(v.dtype)
                    out = torch.matmul(p, v).transpose(1, 2).reshape(bsz, -1, attn.heads * head_dim)
                    out = out.to(q.dtype)
                out = attn.to_out[0](out)
                out = attn.to_out[1](out)
                if not self.cfg.force_orig_all:
                    cache["prev_q"], cache["prev_k"], cache["prev_scores"] = q.detach(), k.detach(), scores_raw.detach()
                return out

            use_diff = self.cfg.enable_attention_ditto and (ctrl.step_idx >= 1) and not self.cfg.force_orig_all
            if not use_diff:
                return _orig_single()

            prev_q, prev_k, prev_scores = cache.get("prev_q"), cache.get("prev_k"), cache.get("prev_scores")
            if (prev_q is None) or (prev_k is None) or (prev_scores is None) or (prev_q.shape != q.shape) or (prev_k.shape != k.shape):
                return _orig_single()

            if (ctrl.cfg.quant_mode in ("qdiff", "fake_dynamic")) and self.cfg.delta_int4:
                qmax8 = (2 ** (self.cfg.a_bits - 1)) - 1
                with torch.no_grad():
                    s_q = q.detach().abs().max().clamp(min=1e-8) / qmax8
                dq_codes = torch.clamp(torch.round((q - prev_q) / s_q), -qmax8, qmax8).to(torch.int16)
                out_q = (dq_codes < -8) | (dq_codes > 7)
                dq4 = torch.where(out_q, torch.zeros_like(dq_codes), torch.clamp(dq_codes, -8, 7)).to(q.dtype) * s_q
                dq8 = torch.where(out_q, dq_codes, torch.zeros_like(dq_codes)).to(q.dtype) * s_q

                with torch.no_grad():
                    s_k = k.detach().abs().max().clamp(min=1e-8) / qmax8
                dk_codes = torch.clamp(torch.round((k - prev_k) / s_k), -qmax8, qmax8).to(torch.int16)
                out_k = (dk_codes < -8) | (dk_codes > 7)
                dk4 = torch.where(out_k, torch.zeros_like(dk_codes), torch.clamp(dk_codes, -8, 7)).to(k.dtype) * s_k
                dk8 = torch.where(out_k, dk_codes, torch.zeros_like(dk_codes)).to(k.dtype) * s_k

                scores_raw = prev_scores + (
                    torch.matmul(prev_q, dk4.transpose(-2, -1)) +
                    torch.matmul(prev_q, dk8.transpose(-2, -1)) +
                    torch.matmul(dq4,    k.transpose(-2, -1)) +
                    torch.matmul(dq8,    k.transpose(-2, -1))
                ) * scale
            else:
                dq = q - prev_q
                dk = k - prev_k
                scores_raw = prev_scores + (torch.matmul(prev_q, dk.transpose(-2, -1)) +
                                            torch.matmul(dq,     k.transpose(-2, -1))) * scale

            scores = scores_raw
            if attention_mask is not None:
                scores = scores + attention_mask
            if getattr(self.cfg, "softmax_offload", False):
                out4 = _softmax_cpu_stream_matmul(scores, v, getattr(self.cfg, "softmax_chunk", 2048))
                out = out4.transpose(1, 2).reshape(bsz, -1, attn.heads * head_dim)
                out = out.to(q.dtype)
            else:
                p = scores.float().softmax(dim=-1).to(v.dtype)
                out = torch.matmul(p, v).transpose(1, 2).reshape(bsz, -1, attn.heads * head_dim)
                out = out.to(q.dtype)
            out = attn.to_out[0](out)
            out = attn.to_out[1](out)

            new_scores_raw = torch.matmul(q, k.transpose(-2, -1)) * scale
            cache["prev_q"], cache["prev_k"], cache["prev_scores"] = q.detach(), k.detach(), new_scores_raw.detach()
            return out

        # Joint attention (sample + context)
        bsz = encoder_hidden_states.shape[0]
        q_s = attn.to_q(hidden_states); k_s = attn.to_k(hidden_states); v_s = attn.to_v(hidden_states)
        q_c = attn.add_q_proj(encoder_hidden_states); k_c = attn.add_k_proj(encoder_hidden_states); v_c = attn.add_v_proj(encoder_hidden_states)

        q = torch.cat([q_s, q_c], dim=1)
        k = torch.cat([k_s, k_c], dim=1)
        v = torch.cat([v_s, v_c], dim=1)

        inner_dim = k.shape[-1]
        head_dim = inner_dim // attn.heads
        q = q.view(bsz, -1, attn.heads, head_dim).transpose(1, 2)
        k = k.view(bsz, -1, attn.heads, head_dim).transpose(1, 2)
        v = v.view(bsz, -1, attn.heads, head_dim).transpose(1, 2)
        scale = head_dim ** -0.5
        n_q_sample = residual.shape[1]

        def _orig_joint():
            scores_raw = torch.matmul(q, k.transpose(-2, -1)) * scale
            scores = scores_raw
            if attention_mask is not None:
                scores = scores + attention_mask
            if getattr(self.cfg, "softmax_offload", False):
                out4 = _softmax_cpu_stream_matmul(scores, v, getattr(self.cfg, "softmax_chunk", 2048))
                hs = out4.transpose(1, 2).reshape(bsz, -1, attn.heads * head_dim)
                hs = hs.to(q.dtype)
            else:
                p = scores.float().softmax(dim=-1).to(v.dtype)
                hs = torch.matmul(p, v).transpose(1, 2).reshape(bsz, -1, attn.heads * head_dim)
                hs = hs.to(q.dtype)
            hs_s, hs_c = hs[:, :n_q_sample], hs[:, n_q_sample:]
            hs_s = attn.to_out[0](hs_s); hs_s = attn.to_out[1](hs_s)
            if not getattr(attn, "context_pre_only", False):
                hs_c = attn.to_add_out(hs_c)
            if not self.cfg.force_orig_all:
                cache["prev_q"], cache["prev_k"], cache["prev_scores"] = q.detach(), k.detach(), scores_raw.detach()
                cache["n_q_sample"] = int(n_q_sample)
            return hs_s, hs_c

        use_diff = self.cfg.enable_attention_ditto and (ctrl.step_idx >= 1) and not self.cfg.force_orig_all
        if not use_diff:
            return _orig_joint()

        prev_q, prev_k, prev_scores = cache.get("prev_q"), cache.get("prev_k"), cache.get("prev_scores")
        prev_n = cache.get("n_q_sample")
        if (prev_q is None) or (prev_k is None) or (prev_scores is None) or (prev_q.shape != q.shape) or (prev_k.shape != k.shape) or (prev_n != n_q_sample):
            return _orig_joint()

        if (ctrl.cfg.quant_mode in ("qdiff", "fake_dynamic")) and self.cfg.delta_int4:
            qmax8 = (2 ** (self.cfg.a_bits - 1)) - 1
            with torch.no_grad():
                s_q = q.detach().abs().max().clamp(min=1e-8) / qmax8
            dq_codes = torch.clamp(torch.round((q - prev_q) / s_q), -qmax8, qmax8).to(torch.int16)
            out_q = (dq_codes < -8) | (dq_codes > 7)
            dq4 = torch.where(out_q, torch.zeros_like(dq_codes), torch.clamp(dq_codes, -8, 7)).to(q.dtype) * s_q
            dq8 = torch.where(out_q, dq_codes, torch.zeros_like(dq_codes)).to(q.dtype) * s_q

            with torch.no_grad():
                s_k = k.detach().abs().max().clamp(min=1e-8) / qmax8
            dk_codes = torch.clamp(torch.round((k - prev_k) / s_k), -qmax8, qmax8).to(torch.int16)
            out_k = (dk_codes < -8) | (dk_codes > 7)
            dk4 = torch.where(out_k, torch.zeros_like(dk_codes), torch.clamp(dk_codes, -8, 7)).to(k.dtype) * s_k
            dk8 = torch.where(out_k, dk_codes, torch.zeros_like(dk_codes)).to(k.dtype) * s_k

            scores_raw = prev_scores + (
                torch.matmul(prev_q, dk4.transpose(-2, -1)) +
                torch.matmul(prev_q, dk8.transpose(-2, -1)) +
                torch.matmul(dq4,    k.transpose(-2, -1)) +
                torch.matmul(dq8,    k.transpose(-2, -1))
            ) * scale
        else:
            dq = q - prev_q
            dk = k - prev_k
            scores_raw = prev_scores + (torch.matmul(prev_q, dk.transpose(-2, -1)) +
                                        torch.matmul(dq,     k.transpose(-2, -1))) * scale

        scores = scores_raw
        if attention_mask is not None:
            scores = scores + attention_mask
        if getattr(self.cfg, "softmax_offload", False):
            out4 = _softmax_cpu_stream_matmul(scores, v, getattr(self.cfg, "softmax_chunk", 2048))
            hs = out4.transpose(1, 2).reshape(bsz, -1, attn.heads * head_dim)
            hs = hs.to(q.dtype)
        else:
            p = scores.float().softmax(dim=-1).to(v.dtype)
            hs = torch.matmul(p, v).transpose(1, 2).reshape(bsz, -1, attn.heads * head_dim)
            hs = hs.to(q.dtype)
        hs_s, hs_c = hs[:, :n_q_sample], hs[:, n_q_sample:]
        hs_s = attn.to_out[0](hs_s); hs_s = attn.to_out[1](hs_s)
        if not getattr(attn, "context_pre_only", False):
            hs_c = attn.to_add_out(hs_c)

        new_scores_raw = torch.matmul(q, k.transpose(-2, -1)) * scale
        cache["prev_q"], cache["prev_k"], cache["prev_scores"] = q.detach(), k.detach(), new_scores_raw.detach()
        cache["n_q_sample"] = int(n_q_sample)
        return hs_s, hs_c


def apply_ditto_to_unet(unet: nn.Module, cfg: DittoConfig):
    for full, child in list(_named_modules_with_types(unet, (nn.Conv2d, nn.Linear))):
        parent_path, leaf = full.rsplit(".", 1) if "." in full else ("", full)
        parent = unet.get_submodule(parent_path) if parent_path else unet
        lid = ("conv" if isinstance(child, nn.Conv2d) else "lin") + f":{full}"

        is_attn_proj = (
            leaf in ("to_q", "to_k", "to_v", "add_q_proj", "add_k_proj", "add_v_proj", "to_add_out")
            or full.endswith(".to_out.0")
        )

        if isinstance(child, nn.Conv2d):
            # Standard Ditto conv wrapper (Δ + quant)
            if not cfg.quantize_attn_only:
                setattr(parent, leaf, DittoConv2d(child, lid, cfg))
            continue

        # Linear
        if is_attn_proj:
            # Quantize projections to W8A8, but NO Δ at layer level
            if (not cfg.quantize_to_q_only) or (leaf == "to_q"):
                setattr(parent, leaf, DittoLinearQuantOnly(child, lid, cfg) if cfg.quant_mode != "fp32" else child)
            else:
                setattr(parent, leaf, child)
        else:
            # Regular Linear: Ditto (Δ + quant) unless user asked for attn-only quantization
            if not cfg.quantize_attn_only:
                setattr(parent, leaf, DittoLinear(child, lid, cfg))
            else:
                setattr(parent, leaf, child)

    # Attention processors unchanged (your existing installation code)
    if DIFFUSERS_AVAILABLE and cfg.enable_attention_ditto and hasattr(unet, "set_attn_processor"):
        try:
            if hasattr(unet, "attn_processors") and isinstance(unet.attn_processors, dict):
                processors = {}
                for name, old_proc in unet.attn_processors.items():
                    cname = old_proc.__class__.__name__ if old_proc is not None else ""
                    use_joint = ("Joint" in cname) or ("FusedJoint" in cname)
                    processors[name] = (
                        DittoJointAttentionProcessor(f"attn:{name}", cfg)
                        if use_joint else DittoAttentionProcessor(f"attn:{name}", cfg)
                    )
                unet.set_attn_processor(processors)
            else:
                processors = {}
                for full, child in _named_modules_with_types(unet, (nn.Module,), ""):
                    if hasattr(child, "to_q") and hasattr(child, "to_k") and hasattr(child, "to_v") and hasattr(child, "to_out"):
                        is_joint = hasattr(child, "add_q_proj") and hasattr(child, "add_k_proj") and hasattr(child, "add_v_proj") and hasattr(child, "to_add_out")
                        processors[full] = (
                            DittoJointAttentionProcessor(f"attn:{full}", cfg)
                            if is_joint else DittoAttentionProcessor(f"attn:{full}", cfg)
                        )
                if processors:
                    unet.set_attn_processor(processors)
        except Exception as e:
            print("[Ditto] set_attn_processor failed, skipping attention override:", repr(e))

def attach_ditto(pipeline: Any, cfg: Optional[DittoConfig] = None) -> DittoController:
    if cfg is None:
        cfg = DittoConfig()
    controller = DittoController(cfg)
    controller.attach(pipeline)
    target = pipeline.unet if hasattr(pipeline, "unet") else getattr(pipeline, "transformer", None)
    if target is None:
        raise AttributeError("attach_ditto: pipeline has neither .unet nor .transformer")
    apply_ditto_to_unet(target, cfg)
    if cfg.print_layer_choices:
        print("[Ditto] Step0: orig; Step1+: diff; DEFO selects per-layer afterwards.")
    return controller
