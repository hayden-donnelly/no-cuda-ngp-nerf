# no-cuda-ngp-nerf
An educational implementation of Instant NGP NeRF using JAX and Numba instead of custom CUDA kernels.

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

## Citations
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