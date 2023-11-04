import jax
import jax.numpy as jnp
import numpy as np
import numba
from devices import cpus, gpus
import matplotlib.pyplot as plt
from flax.training.train_state import TrainState
from dataset import Dataset, process_3x4_transform_matrix
from typing import Optional
import os

def alpha_composite(foreground, background, alpha):
    return foreground + background * (1 - alpha)

def render_pixel(densities:jax.Array, colors:jax.Array, z_vals:jax.Array, directions:jax.Array):  
    eps = 1e-10
    deltas = jnp.concatenate([
        z_vals[1:] - z_vals[:-1], 
        jnp.broadcast_to(1e10, z_vals[:1].shape)
    ], axis=-1)
    deltas = jnp.expand_dims(deltas, axis=-1)
    deltas = deltas * jnp.linalg.norm(directions, keepdims=True, axis=-1)
    alphas = 1.0 - jnp.exp(-densities * deltas)
    accum_prod = jnp.concatenate([
        jnp.ones_like(alphas[:1], alphas.dtype),
        jnp.cumprod(1.0 - alphas[:-1] + eps, axis=0)
    ], axis=0)
    weights = alphas * accum_prod
    rendered_color = jnp.sum(weights * colors, axis=0)
    accumulated_alpha = jnp.sum(weights, axis=0)
    return rendered_color, accumulated_alpha

@jax.jit
def get_ray(uv_x, uv_y, transform_matrix, c_x, c_y, fl_x, fl_y):
    direction = jnp.array([(uv_x - c_x) / fl_x, (uv_y - c_y) / fl_y, 1.0])
    direction = transform_matrix[:3, :3] @ direction
    direction = direction / jnp.linalg.norm(direction)
    origin = transform_matrix[:3, -1]
    return origin, direction

@numba.njit
def trace_ray(ray_origin, ray_direction, z_step, num_samples_allowed):    
    box_min = 0.0
    box_max = 1.0
    num_valid_samples = 0
    z = 0.0
    ray_far = 3.0
    has_entered_box = False
    
    valid_z_vals = np.zeros((num_samples_allowed,), dtype=np.float32)
    valid_samples = np.zeros((num_samples_allowed, 3), dtype=np.float32)
    matching_directions = np.zeros((num_samples_allowed, 3))

    while z < ray_far and num_valid_samples < num_samples_allowed:
        current_sample = [
            ray_origin[0] + ray_direction[0] * z, 
            ray_origin[1] + ray_direction[1] * z, 
            ray_origin[2] + ray_direction[2] * z
        ]
        x_check = current_sample[0] >= box_min and current_sample[0] <= box_max
        y_check = current_sample[1] >= box_min and current_sample[1] <= box_max
        z_check = current_sample[2] >= box_min and current_sample[2] <= box_max
        if x_check and y_check and z_check:
            has_entered_box = True
            valid_z_vals[num_valid_samples] = z
            valid_samples[num_valid_samples] = current_sample
            matching_directions[num_valid_samples] = [
                ray_direction[0], ray_direction[1], ray_direction[2]
            ]
            num_valid_samples += 1
        elif has_entered_box:
            break
        z += z_step

    valid_z_vals = valid_z_vals[:num_valid_samples]
    valid_samples = valid_samples[:num_valid_samples]
    matching_directions = matching_directions[:num_valid_samples]
    return valid_samples, matching_directions, valid_z_vals, num_valid_samples

@numba.njit
def batch_trace_rays(ray_origins, ray_directions, batch_size):
    batch_samples = np.zeros((batch_size, 3), dtype=np.float32)
    batch_directions = np.zeros((batch_size, 3), dtype=np.float32)
    batch_z_vals = np.zeros((batch_size,))
    ray_start_indices = np.zeros((batch_size,), dtype=np.int32)

    ray_index = 0
    num_samples = 0
    num_new_samples_allowed = batch_size
    z_step = np.sqrt(3) / 128
    while num_new_samples_allowed > 0:
        new_samples, matching_directions, new_z_vals, num_new_samples = trace_ray(
            ray_origins[ray_index],
            ray_directions[ray_index],
            z_step,
            num_new_samples_allowed
        )
        ray_start_indices[ray_index] = num_samples
        updated_num_samples = num_samples + num_new_samples
        batch_z_vals[num_samples:updated_num_samples] = new_z_vals
        batch_samples[num_samples:updated_num_samples] = new_samples
        batch_directions[num_samples:updated_num_samples] = matching_directions
        num_samples = updated_num_samples

        ray_index += 1
        num_new_samples_allowed -= num_new_samples

    return (
        batch_samples, batch_directions, batch_z_vals, 
        ray_start_indices[:ray_index], num_samples, ray_index
    )

# Batch ray tracing but with batch size bounded by max_samples_per_ray and num_rays.
@numba.njit
def batch_trace_rays_bounded(ray_origins, ray_directions, max_samples_per_ray, num_rays):
    batch_size = max_samples_per_ray * num_rays
    batch_samples = np.zeros((batch_size, 3), dtype=np.float32)
    batch_directions = np.zeros((batch_size, 3), dtype=np.float32)
    batch_z_vals = np.zeros((batch_size,))
    ray_start_indices = np.zeros((batch_size,), dtype=np.int32)

    ray_index = 0
    num_samples = 0
    z_step = np.sqrt(3) / 128
    while ray_index < num_rays:
        new_samples, matching_directions, new_z_vals, num_new_samples = trace_ray(
            ray_origins[ray_index],
            ray_directions[ray_index],
            z_step,
            max_samples_per_ray
        )
        ray_start_indices[ray_index] = num_samples
        updated_num_samples = num_samples + num_new_samples
        batch_z_vals[num_samples:updated_num_samples] = new_z_vals
        batch_samples[num_samples:updated_num_samples] = new_samples
        batch_directions[num_samples:updated_num_samples] = matching_directions
        num_samples = updated_num_samples
        ray_index += 1

    return (
        batch_samples, batch_directions, batch_z_vals, 
        ray_start_indices[:ray_index], num_samples, ray_index
    )

@numba.njit
def batch_render(densities, colors, z_vals, ray_start_indices, num_rays):
    rendered_colors = np.zeros((num_rays, 3), dtype=np.float32)
    rendered_alphas = np.zeros((num_rays, 1), dtype=np.float32)
    rendered_depths = np.zeros((num_rays, 1), dtype=np.float32)
    for i in range(num_rays):
        start_index = ray_start_indices[i]
        end_index = ray_start_indices[i+1] if i < num_rays - 1 else ray_start_indices[-1]
        slice_size = end_index - start_index
        if start_index == end_index:
            continue
    
        current_densities = np.ravel(densities[start_index:end_index])
        current_colors = colors[start_index:end_index]
        current_z_vals = z_vals[start_index:end_index]

        deltas = np.zeros((slice_size,), dtype=np.float32)
        deltas[:-1] = current_z_vals[1:] - current_z_vals[:-1]
        #deltas[-1] = 1e10
        deltas[-1] = 1e-10
        alphas = 1.0 - np.exp(-current_densities * deltas)
        transmittances = np.zeros((slice_size,), dtype=np.float32)
        transmittances[0] = 1.0
        transmittances[1:] = np.cumprod(1.0 - alphas[:-1] + 1e-10)
        weights = alphas * transmittances
        expanded_weights = np.expand_dims(weights, axis=-1)
        rendered_colors[i] = np.sum(expanded_weights * current_colors, axis=0)
        rendered_alphas[i] = np.sum(expanded_weights, axis=0)
        rendered_depths[i] = np.sum(weights * current_z_vals, axis=0)
    return rendered_colors, rendered_alphas, rendered_depths

@numba.njit
def batch_render_backward(
    raw_densities, raw_colors, z_vals, ray_start_indices, num_rays, batch_size, 
    rendered_colors, rendered_alphas, rendered_color_grads, rendered_alpha_grads,
):
    # Output: 
    # raw_density_grads [batch_size, 1]
    # raw_color_grads [batch_size, 3], 
    # z_val_grads [batch_size,]
    # Only the valid ray samples should have grads, the rest should be zero.
    raw_density_grads = np.zeros((batch_size, 1), dtype=np.float32)
    raw_color_grads = np.zeros((batch_size, 3), dtype=np.float32)
    z_val_grads = np.zeros((batch_size,), dtype=np.float32)

    for i in range(num_rays):
        start_index = ray_start_indices[i]
        end_index = ray_start_indices[i+1] if i < num_rays - 1 else ray_start_indices[-1]
        slice_size = end_index - start_index
        if start_index == end_index:
            continue
        
        ray_raw_densities = np.ravel(raw_densities[start_index:end_index])
        ray_raw_colors = raw_colors[start_index:end_index]
        ray_z_vals = z_vals[start_index:end_index]
        ray_rendered_color = rendered_colors[i]
        ray_rendered_alpha = rendered_alphas[i]

        deltas = np.zeros((slice_size,), dtype=np.float32)
        deltas[:-1] = ray_z_vals[1:] - ray_z_vals[:-1]
        #deltas[-1] = 1e10
        deltas[-1] = 1e-10
        alphas = 1.0 - np.exp(-ray_raw_densities * deltas)
        transmittances = np.zeros((slice_size,), dtype=np.float32)
        transmittances[0] = 1.0
        transmittances[1:] = np.cumprod(1.0 - alphas[:-1] + 1e-10)
        weights = alphas * transmittances
        weights = np.expand_dims(weights, axis=-1)
        weighted_raw_colors = weights * ray_raw_colors

        dray_rendered_color_dray_raw_densities = (
            np.expand_dims(deltas, axis=-1) * (
                ray_raw_colors * np.expand_dims(transmittances, axis=-1) 
                - (np.expand_dims(ray_rendered_color, axis=0) - weighted_raw_colors)
            )
        )
        drendered_alpha_draw_densities = np.expand_dims(
            deltas * (1 - ray_rendered_alpha), axis=-1
        )
        dl_dray_rendered_color = np.expand_dims(rendered_color_grads[i], axis=0)
        dl_dray_rendered_alpha = np.expand_dims(rendered_alpha_grads[i], axis=0)
        # dl/dsigma = dl/dc * dc/dsigma + dl/dalpha * dalpha/dsigma
        dl_dray_raw_densities = (
            # dl/dc * dc/dsigma
            dl_dray_rendered_color[0, 0] * dray_rendered_color_dray_raw_densities[:, 0:1]
            + dl_dray_rendered_color[0, 1] * dray_rendered_color_dray_raw_densities[:, 1:2]
            + dl_dray_rendered_color[0, 2] * dray_rendered_color_dray_raw_densities[:, 2:3]
            # dl/dalpha * dalpha/dsigma
            + dl_dray_rendered_alpha * drendered_alpha_draw_densities
        )
        raw_density_grads[start_index:end_index] = dl_dray_raw_densities
        raw_color_grads[start_index:end_index] = weights * rendered_color_grads[i]

    return raw_density_grads, raw_color_grads, z_val_grads

@jax.custom_vjp
def differentiable_render(
    densities, colors, z_vals, ray_start_indices, num_rays
):
    cpu_densities = np.array(jax.device_put(densities, cpus[0]))
    cpu_colors = np.array(jax.device_put(colors, cpus[0]))
    cpu_ray_start_indices = np.array(jax.device_put(ray_start_indices, cpus[0]))
    cpu_z_vals = np.array(jax.device_put(z_vals, cpus[0]))

    rendered_colors, rendered_alphas, _ = batch_render(
        cpu_densities, cpu_colors, cpu_z_vals, cpu_ray_start_indices, num_rays
    )
    rendered_colors = jnp.array(rendered_colors)
    rendered_alphas = jnp.array(rendered_alphas)
    return rendered_colors, rendered_alphas

def differentiable_render_fwd(
    raw_densities, raw_colors, z_vals, ray_start_indices, num_rays
):
    # Feels a bit weird to have the forward pass wrap the primal function
    # when the backward pass skips the primal function entirely.
    # I wonder if I can write the primal function inline here since the backward
    # pass doesn't need it.
    primal_outputs = differentiable_render(
        raw_densities, raw_colors, z_vals, ray_start_indices, num_rays
    )
    rendered_colors, rendered_alphas = primal_outputs
    residuals = (
        rendered_colors, rendered_alphas,
        raw_densities, raw_colors, z_vals,
        ray_start_indices, num_rays
    )
    return primal_outputs, residuals

def differentiable_render_bwd(residuals, gradients):
    rendered_color_grads, rendered_alpha_grads = gradients
    (
        rendered_colors, rendered_alphas,
        raw_densities, raw_colors, z_vals,
        ray_start_indices, num_rays
    ) = residuals

    cpu_raw_densities = np.array(jax.device_put(raw_densities, cpus[0]))
    cpu_raw_colors = np.array(jax.device_put(raw_colors, cpus[0]))
    cpu_z_vals = np.array(jax.device_put(z_vals, cpus[0]))
    cpu_ray_start_indices = np.array(jax.device_put(ray_start_indices, cpus[0]))
    cpu_rendered_colors = np.array(jax.device_put(rendered_colors, cpus[0]))
    cpu_rendered_alphas = np.array(jax.device_put(rendered_alphas, cpus[0]))
    cpu_rendered_color_grads = np.array(jax.device_put(rendered_color_grads, cpus[0]))
    cpu_rendered_alpha_grads = np.array(jax.device_put(rendered_alpha_grads, cpus[0]))

    raw_density_grads, raw_color_grads, z_val_grads = batch_render_backward(
        cpu_raw_densities, cpu_raw_colors, cpu_z_vals, 
        cpu_ray_start_indices, num_rays, int(cpu_raw_densities.shape[0]),
        cpu_rendered_colors, cpu_rendered_alphas,
        cpu_rendered_color_grads, cpu_rendered_alpha_grads
    )
    raw_density_grads = jax.device_put(jnp.array(raw_density_grads), gpus[0])
    raw_color_grads = jax.device_put(jnp.array(raw_color_grads), gpus[0])
    # Currently not computing z_val_grads, so it's always None. 
    z_val_grads = None
    #z_val_grads = jax.device_put(jnp.array(z_val_grads), gpus[0])
    return raw_density_grads, raw_color_grads, z_val_grads, None, None

differentiable_render.defvjp(differentiable_render_fwd, differentiable_render_bwd)

def render_scene(
    max_samples_per_ray:int,
    patch_size_x:int,
    patch_size_y:int,
    dataset:Dataset, 
    transform_matrix:jnp.ndarray, 
    state:TrainState,
    file_name:Optional[str]='rendered_image'
):    
    @jax.jit
    def compute_sample(params, ray_sample, direction):
        return state.apply_fn({'params': params}, (ray_sample, direction))
    compute_batch = jax.vmap(compute_sample, in_axes=(None, 0, 0))
    get_ray_vmap = jax.vmap(get_ray, in_axes=(0, 0, None, None, None, None, None))
    cpus = jax.devices("cpu")

    num_patches_x = dataset.w // patch_size_x
    num_patches_y = dataset.h // patch_size_y
    patch_area = patch_size_x * patch_size_y
    image = np.ones((dataset.w, dataset.h, 3), dtype=np.float32)
    depth_map = np.ones((dataset.w, dataset.h, 1), dtype=np.float32)

    for x in range(num_patches_x):
        patch_start_x = patch_size_x * x
        patch_end_x = patch_start_x + patch_size_x
        x_coordinates = jnp.arange(patch_start_x, patch_end_x)
        for y in range(num_patches_y):
            patch_start_y = patch_size_y * y
            patch_end_y = patch_start_y + patch_size_y
            y_coordinates = jnp.arange(patch_start_y, patch_end_y)
            
            x_grid_coordinates, y_grid_coordinates = jnp.meshgrid(x_coordinates, y_coordinates)
            x_grid_coordinates = jnp.ravel(x_grid_coordinates)
            y_grid_coordinates = jnp.ravel(y_grid_coordinates)
            ray_origins, ray_directions = get_ray_vmap(
                x_grid_coordinates, 
                y_grid_coordinates, 
                transform_matrix,
                dataset.cx, 
                dataset.cy, 
                dataset.fl_x, 
                dataset.fl_y
            )

            cpu_ray_origins = np.array(jax.device_put(ray_origins, cpus[0]))
            cpu_ray_directions = np.array(jax.device_put(ray_directions, cpus[0]))
            (batch_samples, batch_directions, batch_z_vals, 
            ray_start_indices, num_valid_samples, num_valid_rays) = \
                batch_trace_rays_bounded(
                    cpu_ray_origins, cpu_ray_directions, max_samples_per_ray, patch_area
                )
            
            batch_samples = jnp.array(batch_samples)
            batch_directions = jnp.array(batch_directions)
            batch_z_vals = jnp.array(batch_z_vals)
            raw_densities, raw_colors = compute_batch(
                state.params, batch_samples, batch_directions
            )
            cpu_raw_densities = np.array(jax.device_put(raw_densities, cpus[0]))
            cpu_raw_colors = np.array(jax.device_put(raw_colors, cpus[0]))
            cpu_batch_z_vals = np.array(jax.device_put(batch_z_vals, cpus[0]))
            rendered_colors, rendered_alphas, rendered_depths = batch_render(
                cpu_raw_densities, cpu_raw_colors, 
                cpu_batch_z_vals, ray_start_indices, num_valid_rays
            )
            rendered_colors = alpha_composite(
                rendered_colors, jnp.ones(rendered_colors.shape), rendered_alphas
            )

            image_patch_shape = (patch_end_x - patch_start_x, patch_end_y - patch_start_y, 3)
            image_patch = np.reshape(rendered_colors, image_patch_shape, order='F')
            image[patch_start_x:patch_end_x, patch_start_y:patch_end_y] = image_patch

            depth_patch_shape = (patch_end_x - patch_start_x, patch_end_y - patch_start_y, 1)
            depth_patch = np.reshape(rendered_depths, depth_patch_shape, order='F')
            depth_map[patch_start_x:patch_end_x, patch_start_y:patch_end_y] = depth_patch

    image = np.nan_to_num(image)
    image = np.clip(image, 0, 1)
    image = np.transpose(image, (1, 0, 2))
    plt.imsave(os.path.join('data/', file_name + '.png'), image)
    depth_map = np.nan_to_num(depth_map)
    depth_map = np.clip(depth_map, 0, 1)
    depth_map = np.transpose(depth_map, (1, 0, 2))
    depth_map = np.squeeze(depth_map, axis=-1)
    plt.imsave(os.path.join('data/', file_name + '_depth.png'), depth_map, cmap='gray')

def turntable_render(
    num_frames:int, 
    max_samples_per_ray:int,
    patch_size_x:int,
    patch_size_y:int,
    camera_distance:float, 
    ray_near:float, 
    ray_far:float, 
    state:TrainState, 
    dataset:Dataset,
    file_name:str='turntable_render'
):
    xy_start_position = jnp.array([0.0, -1.0])
    xy_start_position_angle_2d = 0
    z_start_rotation_angle_3d = 0
    angle_delta = 2 * jnp.pi / num_frames

    x_rotation_angle_3d = jnp.pi / 2
    x_rotation_matrix = jnp.array([
        [1, 0, 0],
        [0, jnp.cos(x_rotation_angle_3d), -jnp.sin(x_rotation_angle_3d)],
        [0, jnp.sin(x_rotation_angle_3d), jnp.cos(x_rotation_angle_3d)],
    ])

    for i in range(num_frames):
        xy_position_angle_2d = xy_start_position_angle_2d + i * angle_delta
        z_rotation_angle_3d = z_start_rotation_angle_3d + i * angle_delta

        xy_rotation_matrix_2d = jnp.array([
            [jnp.cos(xy_position_angle_2d), -jnp.sin(xy_position_angle_2d)], 
            [jnp.sin(xy_position_angle_2d), jnp.cos(xy_position_angle_2d)]
        ])
        current_xy_position = xy_rotation_matrix_2d @ xy_start_position
    
        z_rotation_matrix = jnp.array([
            [jnp.cos(z_rotation_angle_3d), -jnp.sin(z_rotation_angle_3d), 0],
            [jnp.sin(z_rotation_angle_3d), jnp.cos(z_rotation_angle_3d), 0],
            [0, 0, 1],
        ])

        rotation_matrix = z_rotation_matrix @ x_rotation_matrix
        translation_matrix = jnp.array([
            [current_xy_position[0]],
            [current_xy_position[1]],
            [0],
        ])
        transform_matrix = jnp.concatenate([rotation_matrix, translation_matrix], axis=-1)
        transform_matrix = process_3x4_transform_matrix(transform_matrix, camera_distance)

        render_scene(
            max_samples_per_ray=max_samples_per_ray, 
            patch_size_x=patch_size_x, 
            patch_size_y=patch_size_y, 
            dataset=dataset, 
            transform_matrix=transform_matrix, 
            state=state,
            file_name=file_name + f'_frame_{i}'
        )