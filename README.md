# ComfyUI Advanced Tiling

[ComfyUI](https://github.com/comfyanonymous/ComfyUI) tiling nodes inspired by [spinagon/ComfyUI-seamless-tiling](https://github.com/spinagon/ComfyUI-seamless-tiling). This project enables the creation of tileable images in various shapes with customizable rotations. Example workflows are located in the `workflows` directory.

![Hexagon tiling example](_media/hexagon.png)

⚠️ There are some artifacts present. I'm not sure if this is because of a flawed implementation or simply because Stable Diffusion isn't intended for this purpose.

## Implemented tiling modes

- [x] Hexagon
- [x] Rectangular (toroidal — right edge wraps to left, bottom wraps to top)
- [x] None (normal generation)

## Supported models

### UNet-based

- [x] Stable Diffusion 1.5 (also 1.4)
- [x] Stable Diffusion 2.1 (also 2.0)
- [x] Stable Diffusion XL (SDXL)

### DiT-based (via toroidal attention patching)

- [x] FLUX.2 — tested on flux.2-klein-4b
- [x] Qwen Image — tested on qwen-image-2512 (Q6 GGUF and fp16)
- [ ] Z-Image / Z-Image Turbo — works with artifacts; wrapping not always consistent, especially at lower scales (~0.87). Less artifacts at higher scales (~0.95).

DiT models are automatically detected and use toroidal attention instead of latent padding for seamless tiling. Lumina/Z-Image models use latent wrapping only (toroidal attention causes boundary noise with multiplicative RoPE).

## Credits

[spinagon/ComfyUI-seamless-tiling](https://github.com/spinagon/ComfyUI-seamless-tiling) [GPL-3.0] - Used as a base for this project

Red Blob Games - [Hexagonal Grids](https://www.redblobgames.com/grids/hexagons/) - Hexagonal grid math
