import flax.linen as nn
from flax.training.train_state import TrainState
import jax
import jax.numpy as jnp
import optax
from dataset import Dataset
from rendering import differentiable_render, get_ray, batch_trace_rays, alpha_composite

def create_train_state(
    model:nn.Module, rng, learning_rate:float, epsilon:float, weight_decay_coefficient:float
):
    x = (jnp.ones([3]) / 3, jnp.ones([3]) / 3)
    variables = model.init(rng, x)
    params = variables['params']
    adam = optax.adam(learning_rate, eps=epsilon, eps_root=epsilon)
    # To prevent divergence after long training periods, the paper applies a weak 
    # L2 regularization to the network weights, but not the hash table entries.
    weight_decay_mask = dict({
        key:True if key != 'MultiResolutionHashEncoding_0' else False
        for key in params.keys()
    })
    weight_decay = optax.add_decayed_weights(weight_decay_coefficient, mask=weight_decay_mask)
    tx = optax.chain(adam, weight_decay)
    ts = TrainState.create(apply_fn=model.apply, params=params, tx=tx)
    return ts

def sample_pixels(rng, num_samples:int, image_width:int, image_height:int, num_images:int):
    width_rng, height_rng, image_rng = jax.random.split(rng, num=3) 
    width_indices = jax.random.randint(
        width_rng, shape=(num_samples,), minval=0, maxval=image_width
    )
    height_indices = jax.random.randint(
        height_rng, shape=(num_samples,), minval=0, maxval=image_height
    )
    image_indices = jax.random.randint(
        image_rng, shape=(num_samples,), minval=0, maxval=num_images
    )
    indices = (image_indices, width_indices, height_indices)
    return indices 

def train_loop(
    batch_size:int, max_num_rays:int, training_steps:int, state:TrainState, dataset:Dataset
):
    @jax.jit
    def compute_sample(params, ray_sample, direction):
        return state.apply_fn({'params': params}, (ray_sample, direction))
    compute_batch = jax.vmap(compute_sample, in_axes=(None, 0, 0))

    get_ray_vmap = jax.vmap(get_ray, in_axes=(0, 0, 0, None, None, None, None))
    cpus = jax.devices("cpu")

    for step in range(training_steps):
        pixel_sample_key, random_bg_key = jax.random.split(jax.random.PRNGKey(step), num=2)
        image_indices, width_indices, height_indices = sample_pixels(
            pixel_sample_key, max_num_rays, dataset.w, dataset.h, dataset.images.shape[0],
        )
        ray_origins, ray_directions = get_ray_vmap(
            width_indices, height_indices, dataset.transform_matrices[image_indices], 
            dataset.cx, dataset.cy, dataset.fl_x, dataset.fl_y
        )
        
        cpu_ray_origins = jax.device_put(ray_origins, cpus[0])
        cpu_ray_directions = jax.device_put(ray_directions, cpus[0])
        
        (batch_samples, batch_directions, batch_z_vals,
        ray_start_indices, num_valid_samples, num_valid_rays) = \
            batch_trace_rays(cpu_ray_origins, cpu_ray_directions, batch_size)

        batch_samples = jnp.array(batch_samples)
        batch_directions = jnp.array(batch_directions)
        batch_directions_norms = jnp.linalg.norm(batch_directions, keepdims=True, axis=-1)
        normalized_batch_directions = batch_directions / batch_directions_norms
        batch_z_vals = jnp.array(batch_z_vals)
        ray_start_indices = jnp.array(ray_start_indices)

        target_pixels = dataset.images[image_indices, height_indices, width_indices]
        target_pixels = target_pixels[:num_valid_rays]

        def loss_fn(params):
            densities, colors = compute_batch(
                params, batch_samples, normalized_batch_directions
            )
            target_colors = target_pixels[:, :3]
            target_alphas = target_pixels[:, 3:]
            #random_bg_colors = jax.random.uniform(random_bg_key, target_colors.shape)
            random_bg_colors = jnp.ones(target_colors.shape)
            rendered_colors, rendered_alphas = differentiable_render(
                densities, colors, batch_z_vals, 
                ray_start_indices, num_valid_rays
            )
            target_colors = alpha_composite(target_colors, random_bg_colors, target_alphas)
            rendered_colors = alpha_composite(rendered_colors, random_bg_colors, rendered_alphas)
            loss = jnp.mean((rendered_colors - target_colors)**2)
            return loss
        
        grad_fn = jax.value_and_grad(loss_fn)
        loss, grads = grad_fn(state.params)
        # The gradients for all invalid samples are zero. 
        # The presence of so many zeros introduces numerical instability which 
        # causes there to be NaNs in the gradients.
        # nan_to_num is a quick way to fix this by setting all the NaNs to zero.
        grads = jax.tree_util.tree_map(jnp.nan_to_num, grads)
        state = state.apply_gradients(grads=grads)
        print('Current step:', step)
        print('Loss:', loss)
        print('Num samples:', num_valid_samples)
        print('Num rays:', num_valid_rays)
        print('Num samples per ray:', num_valid_samples / num_valid_rays)
    return state