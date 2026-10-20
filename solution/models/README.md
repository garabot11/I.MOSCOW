# Trained models (owner: ml-builder)

Copied into the image as `/workspace/models/` (Dockerfile). Expected: `d3_unet.pt` (TorchScript,
exported with torch 2.11.0+cu128 = the version of the image), `d3_unet.state.pt`, `model_card.md`.
`metro doctor` lists the `*.pt` files it finds here.
