import flax.linen as nn
import jax.numpy as jnp

# Calculates the fourth order spherical harmonic encoding for the given direction.
# The order is always 4, so num_components is always 16 (order^2).
# This is hardcoded because no other order of spherical harmonics is used.
def fourth_order_sh_encoding(direction:jnp.ndarray):
    x = direction[0]
    y = direction[1]
    z = direction[2]

    xx = x**2
    yy = y**2
    zz = z**2

    components = jnp.array([
        0.28209479177387814,
        0.4886025119029199 * y,
        0.4886025119029199 * z,
        0.4886025119029199 * x,
        1.0925484305920792 * x * y,
        1.0925484305920792 * y * z,
        0.9461746957575601 * zz - 0.31539156525251999,
        1.0925484305920792 * x * z,
        0.5462742152960396 * (xx - yy),
        0.5900435899266435 * y * (3 * xx - yy),
        2.890611442640554 * x * y * z,
        0.4570457994644658 * y * (5 * zz - 1),
        0.3731763325901154 * z * (5 * zz - 3),
        0.4570457994644658 * x * (5 * zz - 1),
        1.445305721320277 * z * (xx - yy),
        0.5900435899266435 * x * (xx - 3 * yy)
    ])
    return components

class MultiResolutionHashEncoding(nn.Module):
    table_size: int
    num_levels: int
    min_resolution: int
    max_resolution: int
    feature_dim: int

    def setup(self):
        self.levels = jnp.arange(self.num_levels)
        self.hash_offset = self.levels * self.table_size
        self.spatial_dim = 3
        if self.num_levels > 1:
            self.growth_factor = jnp.exp(
                (jnp.log(self.max_resolution) - jnp.log(self.min_resolution)) 
                / (self.num_levels - 1)
            )
        else:
            self.growth_factor = 1
        self.scalings = jnp.floor(self.min_resolution * self.growth_factor**self.levels)
        self.scalings = jnp.reshape(self.scalings, (self.scalings.shape[0], 1))
        absolute_table_size = self.table_size * self.num_levels
        # Feature dim comes first so features can be broadcast with point offset.
        # Feature shape is (feature_dim, num_levels). 
        # Point offset shape is (spatial_dim, num_levels).
        self.hash_table = self.param(
            'hash_table', 
            nn.initializers.uniform(scale=10**-4), 
            (self.feature_dim, absolute_table_size)
        )

    def hash_function(self, x:jnp.ndarray, table_size:int, hash_offset:jnp.ndarray):
        pre_xor = x * jnp.array([1, 2654435761, 805459861])
        x = jnp.bitwise_xor(pre_xor[:, 0], pre_xor[:, 1])
        x = jnp.bitwise_xor(x, pre_xor[:, 2])
        x = x % table_size
        x = x + hash_offset
        return x
    
    def __call__(self, x:jnp.ndarray):
        scaled = x * self.scalings
        scaled_c = jnp.ceil(scaled).astype(jnp.int32)
        scaled_f = jnp.floor(scaled).astype(jnp.int32)
        point_offset = jnp.reshape(scaled - scaled_f, (self.spatial_dim, self.num_levels))
        
        vertex_0 = scaled_c
        vertex_1 = jnp.concatenate([scaled_c[:, 0:1], scaled_c[:, 1:2], scaled_f[:, 2:3]], axis=-1)
        vertex_2 = jnp.concatenate([scaled_c[:, 0:1], scaled_f[:, 1:2], scaled_c[:, 2:3]], axis=-1)
        vertex_3 = jnp.concatenate([scaled_f[:, 0:1], scaled_c[:, 1:2], scaled_c[:, 2:3]], axis=-1)
        vertex_4 = jnp.concatenate([scaled_c[:, 0:1], scaled_f[:, 1:2], scaled_f[:, 2:3]], axis=-1)
        vertex_5 = jnp.concatenate([scaled_f[:, 0:1], scaled_c[:, 1:2], scaled_f[:, 2:3]], axis=-1)
        vertex_6 = jnp.concatenate([scaled_f[:, 0:1], scaled_f[:, 1:2], scaled_c[:, 2:3]], axis=-1)
        vertex_7 = jnp.concatenate([scaled_f[:, 0:1], scaled_f[:, 1:2], scaled_f[:, 2:3]], axis=-1)

        hashed_0 = self.hash_function(vertex_0, self.table_size, self.hash_offset)
        hashed_1 = self.hash_function(vertex_1, self.table_size, self.hash_offset)
        hashed_2 = self.hash_function(vertex_2, self.table_size, self.hash_offset)
        hashed_3 = self.hash_function(vertex_3, self.table_size, self.hash_offset)
        hashed_4 = self.hash_function(vertex_4, self.table_size, self.hash_offset)
        hashed_5 = self.hash_function(vertex_5, self.table_size, self.hash_offset)
        hashed_6 = self.hash_function(vertex_6, self.table_size, self.hash_offset)
        hashed_7 = self.hash_function(vertex_7, self.table_size, self.hash_offset)

        f_0 = self.hash_table[:, hashed_0]
        f_1 = self.hash_table[:, hashed_1]
        f_2 = self.hash_table[:, hashed_2]
        f_3 = self.hash_table[:, hashed_3]
        f_4 = self.hash_table[:, hashed_4]
        f_5 = self.hash_table[:, hashed_5]
        f_6 = self.hash_table[:, hashed_6]
        f_7 = self.hash_table[:, hashed_7]

        # Linearly interpolate between all of the features.
        f_03 = f_0 * point_offset[0:1, :] + f_3 * (1 - point_offset[0:1, :])
        f_12 = f_1 * point_offset[0:1, :] + f_2 * (1 - point_offset[0:1, :])
        f_56 = f_5 * point_offset[0:1, :] + f_6 * (1 - point_offset[0:1, :])
        f_47 = f_4 * point_offset[0:1, :] + f_7 * (1 - point_offset[0:1, :])

        f0312 = f_03 * point_offset[1:2, :] + f_12 * (1 - point_offset[1:2, :])
        f4756 = f_47 * point_offset[1:2, :] + f_56 * (1 - point_offset[1:2, :])

        encoded_value = f0312 * point_offset[2:3, :] + f4756 * (
            1 - point_offset[2:3, :]
        )
        # Transpose so that features are contiguous.
        # i.e. [[f_0_x, f_0_y], [f_1_x, f_2_y], ...]
        # Then ravel to get the entire encoding.
        return jnp.ravel(jnp.transpose(encoded_value))

class InstantNerf(nn.Module):
    number_of_grid_levels: int # Corresponds to L in the paper.
    max_hash_table_entries: int # Corresponds to T in the paper.
    hash_table_feature_dim: int # Corresponds to F in the paper.
    coarsest_resolution: int # Corresponds to N_min in the paper.
    finest_resolution: int # Corresponds to N_max in the paper.
    density_mlp_width: int
    color_mlp_width: int
    high_dynamic_range: bool
    exponential_density_activation: bool

    @nn.compact
    def __call__(self, x):
        position, direction = x
        encoded_position = MultiResolutionHashEncoding(
            table_size=self.max_hash_table_entries,
            num_levels=self.number_of_grid_levels,
            min_resolution=self.coarsest_resolution,
            max_resolution=self.finest_resolution,
            feature_dim=self.hash_table_feature_dim
        )(position)
        x = encoded_position

        x = nn.Dense(self.density_mlp_width, kernel_init=nn.initializers.normal())(x)
        x = nn.activation.relu(x)
        x = nn.Dense(16, kernel_init=nn.initializers.normal())(x)
        if self.exponential_density_activation:
            density = jnp.exp(x[0:1])
        else:
            density = nn.activation.relu(x[0:1])
        density_feature = x

        encoded_direction = fourth_order_sh_encoding(direction)
        x = jnp.concatenate([density_feature, jnp.ravel(encoded_direction)], axis=0)
        x = nn.Dense(self.color_mlp_width, kernel_init=nn.initializers.normal())(x)
        x = nn.activation.relu(x)
        x = nn.Dense(self.color_mlp_width, kernel_init=nn.initializers.normal())(x)
        x = nn.activation.relu(x)
        x = nn.Dense(3)(x)

        if self.high_dynamic_range:
            color = jnp.exp(x)
        else:
            color = nn.activation.sigmoid(x)

        return density, color