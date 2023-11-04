from setuptools import setup, find_packages

setup(
    name = 'no-cuda-ngp-nerf',
    packages=['no_cuda_ngp_nerf'],
    version = '0.1.0',
    license='Apache-2.0',
    description = 'An educational implementation of Instant NGP NeRF using JAX and Numba instead of custom CUDA kernels.',
    long_description_content_type = 'text/markdown',
    author = 'Hayden Donnelly',
    author_email = 'donnellyhd@outlook.com',
    url = 'https://github.com/hayden-donnelly/no-cuda-ngp-nerf',
    install_requires=[
        'flax>=0.7.2',
        'pillow>=10.0.1',
        'matplotlib>=3.7.3',
        'numba>=0.58.0'
    ],
    classifiers=[
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3.6',
    ],
)