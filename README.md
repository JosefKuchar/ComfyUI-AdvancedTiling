# ComfyUI Advanced Tiling

[ComfyUI](https://github.com/comfyanonymous/ComfyUI) tiling nodes inspired by [spinagon/ComfyUI-seamless-tiling](https://github.com/spinagon/ComfyUI-seamless-tiling). This project enables the creation of tileable images in various shapes with customizable rotations. Example workflows are located in the `workflows` directory.

![Hexagon tiling example](_media/hexagon.png)

⚠️ There are some artifacts present. I'm not sure if this is because of a flawed implementation or simply because Stable Diffusion isn't intended for this purpose.

## Implemented tiling modes

- [x] Hexagon
- [x] None (normal generation)

## TODO

- More tiling modes
- Optimize VAE decode (first pass is very slow)

## Credits

[spinagon/ComfyUI-seamless-tiling](https://github.com/spinagon/ComfyUI-seamless-tiling) [GPL-3.0] - Used as a base for this project

Red Blob Games - [Hexagonal Grids](https://www.redblobgames.com/grids/hexagons/) - Hexagonal grid math
