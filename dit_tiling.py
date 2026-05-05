"""
DiT model tiling via toroidal attention and latent wrapping.

Applies toroidal attention (K/V injection at boundary patches) combined with
latent wrapping (filling margin/waste positions with content from opposite edges
on each denoising step). The combination provides seamless infinite tiling:
- Toroidal attention makes boundary patches attend to opposite-edge content
- Latent wrapping provides spatial context at the edges for the model

For Hexagon mode:
- Toroidal attention + latent wrapping using hex coordinate mapping
- Waste positions (outside hex) filled with hex source content

For Rectangular mode:
- Toroidal attention + latent wrapping using scale-based margins
- The working rectangle is centered in the latent with margins on all sides
- Margin positions filled with content from opposite edges of working rectangle
- VAE decoder crops output to the working rectangle
"""

import torch
import torch.nn as nn
from torch.nn import Conv2d

from .modes import ResolvedSettings


def _has_conv2d(model: nn.Module) -> bool:
    """Check if the model has any Conv2d layers (UNet vs DiT)."""
    return any(isinstance(m, Conv2d) for m in model.modules())


def _apply_latent_wrapping(x: torch.Tensor, resolved: ResolvedSettings):
    """Fill margin/waste positions with content from opposite edges.

    Relies on calculate_mapping's @functools.cache for memoization.
    """
    from .advanced_tiling import calculate_mapping

    is_5d = x.ndim == 5
    if is_5d:
        _, _, _, H, W = x.shape
    else:
        _, _, H, W = x.shape

    mapping = calculate_mapping((W, H), (W, H), resolved)

    if is_5d:
        x[:, :, :, mapping[1], mapping[0]] = x[:, :, :, mapping[3], mapping[2]]
    else:
        x[:, :, mapping[1], mapping[0]] = x[:, :, mapping[3], mapping[2]]


def _create_content_wrapper(resolved: ResolvedSettings):
    """Create a model function wrapper that applies latent wrapping each step."""
    def wrapper(apply_model, args):
        _apply_latent_wrapping(args["input"], resolved)
        return apply_model(args["input"], args["timestep"], **args["c"])
    wrapper._is_tiling_wrapper = True
    return wrapper


def _create_lumina_wrapper(resolved: ResolvedSettings):
    """Model wrapper for Lumina: applies latent content wrapping on each step."""
    def wrapper(apply_model, args):
        _apply_latent_wrapping(args["input"], resolved)
        return apply_model(args["input"], args["timestep"], **args["c"])
    wrapper._is_tiling_wrapper = True
    return wrapper


def _is_lumina(diff_model) -> bool:
    """Check if the model uses Lumina/NextDiT architecture (rope_embedder instead of pe_embedder)."""
    return hasattr(diff_model, 'rope_embedder') and not hasattr(diff_model, 'pe_embedder')


def _patch_lumina(model_patcher, diff_model, resolved: ResolvedSettings):
    """Set up tiling for Lumina/NextDiT models.

    Two mechanisms work together:
    1. Latent wrapping (model function wrapper): fills margin/waste positions
       with content from opposite edges on each denoising step.
    2. Waste token reset (double_block_patch): resets waste tokens to their
       source content after each transformer block, preventing garbage
       accumulation from polluting attention for working-area patches.
    """
    patch_size = diff_model.patch_size

    # 1. Latent wrapping
    wrapper = _create_lumina_wrapper(resolved)
    model_patcher.set_model_unet_function_wrapper(wrapper)

    # 2. Waste token reset
    from .toroidal_attention import LuminaWastePatch
    waste_patch = LuminaWastePatch(patch_size, resolved)
    model_patcher.set_model_double_block_patch(waste_patch)


def patch_dit_model(model_patcher, resolved: ResolvedSettings):
    """
    Apply tiling to a DiT model.

    Flux-style models (pe_embedder): attn1_patch for K/V injection + latent wrapping.
    Lumina/NextDiT models (rope_embedder): latent wrapping only (K/V injection
    causes boundary noise with multiplicative RoPE).

    For Hexagon mode: uses hex coordinate mapping for waste positions.
    For Rectangular mode: uses scale-based margins centered in the latent.

    :param model_patcher: ComfyUI ModelPatcher instance
    :param resolved: Resolved tiling settings with pre-computed values
    """
    diff_model = model_patcher.model.diffusion_model

    if resolved.mode == "Hexagon":
        if _is_lumina(diff_model):
            _patch_lumina(model_patcher, diff_model, resolved)

        elif hasattr(diff_model, 'pe_embedder'):
            from .toroidal_attention import HexToroidalAttentionPatch

            patch = HexToroidalAttentionPatch(resolved, diff_model.pe_embedder)
            model_patcher.set_model_attn1_patch(patch)

            wrapper = _create_content_wrapper(resolved)
            model_patcher.set_model_unet_function_wrapper(wrapper)

        else:
            raise ValueError(
                "Model does not have pe_embedder or rope_embedder. "
                "Toroidal attention requires a model with RoPE position embeddings."
            )

    elif resolved.mode == "Rectangular":
        if _is_lumina(diff_model):
            _patch_lumina(model_patcher, diff_model, resolved)

        elif hasattr(diff_model, 'pe_embedder'):
            from .toroidal_attention import RectToroidalAttentionPatch

            patch = RectToroidalAttentionPatch(resolved, diff_model.pe_embedder)
            model_patcher.set_model_attn1_patch(patch)

            wrapper = _create_content_wrapper(resolved)
            model_patcher.set_model_unet_function_wrapper(wrapper)

        else:
            raise ValueError(
                "Model does not have pe_embedder or rope_embedder. "
                "Toroidal attention requires a model with RoPE position embeddings."
            )
