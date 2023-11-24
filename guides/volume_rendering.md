# Volume Rendering

WORK IN PROGRESS.

Neural radiance fields, or NeRFs, are functions that represent 3D scenes by mapping coordinates
and viewing angles to densities and colors. They belong to a group of neural networks called 
*neural fields*. You may have also heard of image fields which represent images by mapping UV 
coordinates to colors, and signed distance fields which represent surfaces by mapping 
coordinates to the signed distance of the nearest surface. The key idea of neural radiance 
fields is that we can query them at any single point within their "field", be it a 3D scene, 
an image, or something else, and they will output some value(s) which correspond to it. This 
distinguishes them from other types of neural networks like diffusion models, autoencoders, 
and classifiers which usually take in many points at once, i.e. all the pixels of an image. 

So, we've setup a NeRF, and we can query it at any point, but how do we train it to represent 
a 3D scene? We know that our NeRF is supposed to output densities and colors, but our dataset 
consists of images and viewing angles. How do we bridge this gap in order to perform 
supervised learning? The answer is something called a *differentiable forward map*. There are 
different types of differentiable forward maps for different types of neural fields, but in 
this case we're interested in *volume rendering*. The original NeRF paper as well as the 
Instant NGP paper both use a volume rendering equation introduced by James T. Kajiya and 
Brian P. Von Herzen in their 1984 paper titled "Ray Tracing Volume Densities". This equation 
models the color of a light ray, $C(r)$, which has travelled through the scene from some near 
time, $t_n$, to some far time, $t_f$, as an infinitesimal sum from $t_n$ to $t_f$ of the ray's 
accumulated *transmittance*, $T(t)$, times the scene density, $\sigma(r(t))$, times the scene 
color, $c(r(t), d)$, where $r(t)$ is the position of the ray at time $t$, and $d$ is the
direction of the ray.

```math
$$
C(r) = \int_{t_n}^{t_f}{T(t)\sigma(r(t))c(r(t), d)dt}
    \text{, where } T(t) = \exp(-\int_{t_n}^{t}{\sigma(r(s))ds})
$$
```

Breaking down this equation, we can see that that two of the factors, the scene density, 
$\sigma(r(t))$, and the scene color, $c(r(t))$, are given to us as outputs of our neural 
network. The third factor, $T(t)$, is a bit more complicated, but our neural network still
gives us everything we need as long as we sample it correctly. Basically, the transmittance is 
the fraction of the ray's light that successfully passes through the scene at the point 
$r(t)$, so $T(t)$ is the sum of these fractions from time $t_n$ to time $t$. You can also
think of $T(t)$ as the probability that the light ray will be fully absorbed and therefore
terminate at time $t$.

With the volume rendering equation, we now have a theoretical map between our training data and
the outputs of our NeRF. Our next challenge is to find some way to approximate the integral of
the volume rendering equation. To accomplish this, we'll shoot a ray from our camera into the 
scene, then march along and sample it at various points. These points can then be passed into 
our NeRF along with their corresponding ray angle to get a set of densities and colors for each
of those points. Note that since the network only operates on individual points and viewing 
angles, these inputs will not interact with and affect each other's outputs. Once we have a set
of densities and colors, we can use the following quadrature rule to approximate the integral
of the volume rendering equation.

```math
$$
\hat{C}(r) = \sum_{i = 1}^{N}{T_i(1 - \exp(-\sigma_i\delta_i))c_i}
    \text{, where } T_i = \exp(-\sum_{j = 1}^{i - 1}{\sigma_j\delta_j})
$$
```