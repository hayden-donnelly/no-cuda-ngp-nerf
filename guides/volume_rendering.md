# Volume Rendering

WORK IN PROGRESS.

Neural radiance fields, or NeRFs, are functions that represent 3D scenes by mapping coordinates and viewing angles to densities and colors. They belong to a group of neural networks called *neural fields*. You may have also heard of image fields which represent images by mapping UV coordinates to colors, and signed distance fields which represent surfaces by mapping coordinates to the signed distance of the nearest surface. The key idea of neural radiance fields is that we can query them at any single point within their "field", be it a 3D scene, an image, or something else, and they will output some value(s) which correspond to it. This distinguishes them from other types of neural networks like diffusion models, autoencoders, and classifiers which usually take in many points at once, i.e. all the pixels of an image. 

So, we've setup a NeRF, and we can query it at any point, but how do we train it to represent a 3D scene? We know that our NeRF is supposed to output densities and colors, but our dataset consists of images and viewing angles. How do we bridge this gap in order to perform supervised learning? The answer is something called a *differentiable forward map*. There are different types of differentiable forward maps for different types of neural fields, but in this case we're interested in *volume rendering*. The original NeRF paper as well as the Insant NGP paper both use a volume rendering equation introduced by James T. Kajiya and Brian P. Von Herzen in their 1984 paper titled "Ray Tracing Volume Densities".

```math
$$
C(r) = \int_{t_n}^{t_f}{T(t)\sigma(r(t))c(r(t), d)dt}
    \text{, where } T_i = \exp(-\int_{t_n}^{t}{\sigma(r(s))ds})
$$
```

The basic idea is to shoot a ray from our camera into the scene, then march along and sample it at various points. These points can then be passed into our NeRF along with their corresponding ray angle to get a set of (predicted) densities and colors for each of those points. Note that since the network only operates on individual points and viewing angles, these inputs will not interact with and affect each other's outputs.

```math
$$
\hat{C}(r) = \sum_{i = 1}^{N}{T_i(1 - \exp(-\sigma_i\delta_i))c_i}
    \text{, where } T_i = \exp(-\sum_{j = 1}^{i - 1}{\sigma_j\delta_j})
$$
```