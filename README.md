# no-cuda-ngp-nerf
An educational implementation of Instant NGP NeRF using JAX and Numba instead of custom CUDA kernels.

Currently a work-in-progress.

## Notes 
- Checkout [derivation_of_derivatives.md](./derivation_of_derivatives.md) to see how the volume rendering derivatives are derived.

## Installation
First clone the repository.
```
https://github.com/hayden-donnelly/no-cuda-ngp-nerf.git
```
Then cd into it and install with pip.
```
cd no-cuda-ngp-nerf
```
```
python3 -m pip install -e .
```
The ``-e`` flag will let you edit the project without having to reinstall.

## Development Environment 

### Docker

- To build the docker image, use `docker-compose build`.
- To start the docker container, use `docker-compose up -d`.
- To open a shell inside the container, use `docker-compose exec ngp bash`.
- To open Jupyter Lab inside the container instead of a shell, go to [http://localhost:7070/](http://localhost:7070/).
- To stop the container, use `docker-compose down`.

### Nix
TODO

## TODO
- [ ] Switch to a Nix development environment.
- [ ] Add occupancy grid bitfield.
- [ ] Add screenshot/video of a trained NeRF render.

## Acknowledgement

Kwea123's [video lecture](https://www.youtube.com/live/c2t_C4-Ovss?si=PYRWj1IZP5y0nJms) and PyTorch implementation of [Instant NGP](https://github.com/kwea123/ngp_pl) were both very helpful during the development of this project.

## Citation
```bibtex
@article{mueller2022instant,
    author={Thomas M\"uller and Alex Evans and Christoph Schied and Alexander Keller},
    title={Instant Neural Graphics Primitives with a Multiresolution Hash Encoding},
    journal={ACM Trans. Graph.},
    issue_date={July 2022},
    volume={41},
    number={4},
    month=jul,
    year={2022},
    pages={102:1--102:15},
    articleno={102},
    numpages={15},
    url={https://doi.org/10.1145/3528223.3530127},
    doi={10.1145/3528223.3530127},
    publisher={ACM},
    address={New York, NY, USA},
}
```
