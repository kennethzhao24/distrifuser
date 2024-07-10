## Installation
```bash
docker build -t distrifuser .

docker run -ti --gpus all -v :/workspace/ distrifuser
```

## TO DO List
- [ ] Models
  - [ ] Text-to-Image
    - [x] DiT
    - [ ] PixelArt-alpha/delta
    - [ ] Stable Diffusion V3
  - [ ] Text-to-Video
    - [ ] OpenSora
    - [ ] Latte
- [ ] Enable CUDA Graph



## Benchmark Results

|       |    Patch    | Naive Patch | Tensor Parallemsim |
|------:|:-----------:|:-----------:|:------------------:|
| 1 GPU |     5.59    |             |                    |
| 2 GPU |     3.13    |             |                    |
| 4 GPU |     2.23    |             |        2.924       |



## Comparison


|             | Model | Data Parallelism | Pipeline Parallelism | Tensor Parallelism |
|------------:|:-----:|:----------------:|:--------------------:|:------------------:|
| ParaDiGMS   |  UNet |        Yes       |          No          |         No         |
| Distrifuser |  UNet |        Yes       |          No          |         No         |
|  PipeFusion |  DiT  |        Yes       |          Yes         |         No         |



## Support for OpenSora Model

## Add DiT Model