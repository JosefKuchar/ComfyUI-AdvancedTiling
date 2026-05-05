"""
Raylight integration for DiT tiling.

Provides AdvancedTilingRay node that works with raylight's RAY_ACTORS type,
applying toroidal attention tiling on distributed workers.

Uses manual Ray remote calls instead of @ray_patch to avoid serialization
issues with custom node packages that have non-standard import paths
(e.g. hyphens in directory names that prevent normal Python imports).
"""

try:
    import raylight.comfy_extra_dist.ray_patch_decorator  # noqa: F401
    HAS_RAYLIGHT = True
except ImportError:
    HAS_RAYLIGHT = False

if HAS_RAYLIGHT:
    import ray

    class AdvancedTilingRay:
        """
        Applies tiling to a raylight RAY_ACTORS model.
        Dispatches per-worker patching via Ray remote calls.

        INPUT_IS_LIST=True: latent is a list (one per GPU worker).
        OUTPUT_IS_LIST=(False, True): actors single, settings list.
        """

        INPUT_IS_LIST = True
        OUTPUT_IS_LIST = (False, True)

        # pylint: disable=invalid-name

        @classmethod
        def INPUT_TYPES(cls):
            return {
                "required": {
                    "settings": ("ADVANCED_TILING_SETTINGS",),
                    "ray_actors": ("RAY_ACTORS",),
                    "vae": ("VAE",),
                    "latent": ("LATENT",),
                },
            }

        RETURN_TYPES = ("RAY_ACTORS", "RESOLVED_TILING_SETTINGS")
        RETURN_NAMES = ("ray_actors", "resolved_settings")
        FUNCTION = "run"
        CATEGORY = "conditioning"

        def run(self, settings, ray_actors, vae, latent):
            settings = settings[0]
            ray_actors = ray_actors[0]
            vae = vae[0]
            vae_factor = vae.spacial_compression_decode()

            if settings.mode == "None":
                resolved_list = []
                for lat in latent:
                    H_lat, W_lat = lat["samples"].shape[-2], lat["samples"].shape[-1]
                    img_w = W_lat * vae_factor
                    img_h = H_lat * vae_factor
                    resolved = settings._resolve_auto(False, vae_factor, 2, img_w, img_h)
                    resolved_list.append(resolved)
                return (ray_actors, resolved_list)

            gpu_workers = ray_actors["workers"]

            # Defined inside run() so cloudpickle serializes it as a nested
            # function (by value) instead of by module reference.  Module-level
            # functions get serialized by reference, which requires importing
            # the module by name on the worker -- but the module name is the
            # filesystem path (with a hyphen), causing ModuleNotFoundError.
            #
            # Worker resolves settings with correct patch_size from the model
            # and returns the resolved settings to the master.
            def _patch(model, vae_factor, mode, rotation, scale, min_margin, divisible_by,
                       img_W, img_H):
                import importlib.util
                import os
                import sys

                _node_dir = os.path.dirname(os.path.abspath(__file__))
                _pkg = "ComfyUI_AdvancedTiling"

                if _pkg not in sys.modules:
                    _init = os.path.join(_node_dir, "__init__.py")
                    spec = importlib.util.spec_from_file_location(
                        _pkg, _init,
                        submodule_search_locations=[_node_dir],
                    )
                    mod = importlib.util.module_from_spec(spec)
                    sys.modules[_pkg] = mod
                    spec.loader.exec_module(mod)

                from ComfyUI_AdvancedTiling.dit_tiling import patch_dit_model
                from ComfyUI_AdvancedTiling.modes import Settings
                from ComfyUI_AdvancedTiling.toroidal_attention import (
                    _BaseToroidalAttentionPatch, LuminaWastePatch,
                )
                import gc
                import torch

                # ModelPatcher.set_model_patch APPENDS to a list — it never
                # removes old entries.  Remove only our own accumulated
                # tiling patches so their cached GPU tensors (synthetic PE,
                # boundary indices) are freed before new patches are added
                # for potentially different latent dimensions.
                to = model.model_options.setdefault("transformer_options", {})
                patches = to.setdefault("patches", {})
                attn = patches.get("attn1_patch", [])
                patches["attn1_patch"] = [p for p in attn
                                          if not isinstance(p, _BaseToroidalAttentionPatch)]
                dbl = patches.get("double_block", [])
                patches["double_block"] = [p for p in dbl
                                           if not isinstance(p, LuminaWastePatch)]
                wrapper = model.model_options.get("model_function_wrapper")
                if wrapper is not None and getattr(wrapper, '_is_tiling_wrapper', False):
                    del model.model_options["model_function_wrapper"]
                gc.collect()
                torch.cuda.empty_cache()

                diff_model = model.model.diffusion_model
                patch_size = getattr(diff_model, 'patch_size', 1)
                raw = Settings(mode, rotation, scale, min_margin, divisible_by, conv2d_attention_wrapping=True)
                resolved = raw._resolve_auto(False, vae_factor, patch_size, img_W, img_H)
                patch_dit_model(model, resolved)
                return {
                    "mode": resolved.mode,
                    "rotation": resolved.rotation,
                    "work_img_w": resolved.work_img_w,
                    "work_img_h": resolved.work_img_h,
                    "margin_img_w": resolved.margin_img_w,
                    "work_patch_w": resolved.work_patch_w,
                    "work_patch_h": resolved.work_patch_h,
                    "hex_size_img": resolved.hex_size_img,
                    "hex_size_patch": resolved.hex_size_patch,
                    "img_w": resolved.img_w,
                    "img_h": resolved.img_h,
                    "conv2d_attention_wrapping": resolved.conv2d_attention_wrapping,
                }

            # Build per-worker args: each worker gets its latent's img dimensions
            worker_args = []
            for lat in latent:
                H_lat, W_lat = lat["samples"].shape[-2], lat["samples"].shape[-1]
                worker_args.append((W_lat * vae_factor, H_lat * vae_factor))

            futures = [
                actor.model_function_runner_get_values.remote(
                    _patch, vae_factor, settings.mode, settings.rotation,
                    settings.scale, settings.min_margin, settings.divisible_by,
                    img_W, img_H,
                )
                for actor, (img_W, img_H) in zip(gpu_workers, worker_args)
            ]
            results = ray.get(futures)
            # Reconstruct ResolvedSettings from plain dicts (workers can't
            # serialize the class back across Ray due to the hyphenated module)
            from .modes import ResolvedSettings
            resolved_list = [ResolvedSettings(**d) for d in results]
            return (ray_actors, resolved_list)
