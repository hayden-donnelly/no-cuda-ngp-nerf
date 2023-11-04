import jax
from dataset import load_dataset
from training import create_train_state, train_loop
from rendering import render_scene, turntable_render
from model import InstantNerf

def main():
    print('GPU:', jax.devices('gpu'))

    dataset_path = 'data/lego'
    downscale_factor = 1
    num_hash_table_levels = 16
    max_hash_table_entries = 2**20
    hash_table_feature_dim = 2
    coarsest_resolution = 16
    finest_resolution = 2**19
    density_mlp_width = 64
    color_mlp_width = 64
    high_dynamic_range = False
    exponential_density_activation = False

    learning_rate = 1e-2
    epsilon = 1e-15
    weight_decay_coefficient = 1e-6
    ray_near = 0.2
    ray_far = 3.0
    batch_size = 30000
    train_target_samples_per_ray = 32
    train_max_rays = batch_size // train_target_samples_per_ray
    render_max_samples_per_ray = 128
    training_steps = 200
    num_turntable_render_frames = 3
    turntable_render_camera_distance = 1.4
    render_patch_size_x = 32
    render_patch_size_y = 32
    num_density_grid_points = 32

    assert ray_near < ray_far, 'Ray near must be less than ray far.'

    dataset = load_dataset(dataset_path, downscale_factor)
    print('Horizontal FOV:', dataset.horizontal_fov)
    print('Vertical FOV:', dataset.vertical_fov)
    print('Focal length x:', dataset.fl_x)
    print('Focal length y:', dataset.fl_y)
    print('Principal point x:', dataset.cx)
    print('Principal point y:', dataset.cy)
    print('Image width:', dataset.w)
    print('Image height:', dataset.h)
    print('Images shape:', dataset.images.shape)

    model = InstantNerf(
        number_of_grid_levels=num_hash_table_levels,
        max_hash_table_entries=max_hash_table_entries,
        hash_table_feature_dim=hash_table_feature_dim,
        coarsest_resolution=coarsest_resolution,
        finest_resolution=finest_resolution,
        density_mlp_width=density_mlp_width,
        color_mlp_width=color_mlp_width,
        high_dynamic_range=high_dynamic_range,
        exponential_density_activation=exponential_density_activation
    )
    rng = jax.random.PRNGKey(1)
    state = create_train_state(
        model=model, rng=rng, learning_rate=learning_rate, 
        epsilon=epsilon, weight_decay_coefficient=weight_decay_coefficient
    )
    del rng
    state = train_loop(
        batch_size=batch_size, max_num_rays=train_max_rays, 
        training_steps=training_steps, state=state, dataset=dataset
    )

    turntable_render(
        num_frames=num_turntable_render_frames,
        max_samples_per_ray=render_max_samples_per_ray,
        patch_size_x=render_patch_size_x, 
        patch_size_y=render_patch_size_y, 
        camera_distance=turntable_render_camera_distance, 
        ray_near=ray_near, 
        ray_far=ray_far, 
        state=state, 
        dataset=dataset, 
        file_name='instant_turntable_render'
    )
    render_scene(
        max_samples_per_ray=render_max_samples_per_ray,
        patch_size_x=render_patch_size_x, 
        patch_size_y=render_patch_size_y, 
        dataset=dataset, 
        transform_matrix=dataset.transform_matrices[9], 
        state=state,
        file_name='instant_rendered_image_0'
    )
    render_scene(
        max_samples_per_ray=render_max_samples_per_ray,
        patch_size_x=render_patch_size_x, 
        patch_size_y=render_patch_size_y, 
        dataset=dataset, 
        transform_matrix=dataset.transform_matrices[14], 
        state=state,
        file_name='instant_rendered_image_1'
    )
    render_scene(
        max_samples_per_ray=render_max_samples_per_ray,
        patch_size_x=render_patch_size_x, 
        patch_size_y=render_patch_size_y, 
        dataset=dataset, 
        transform_matrix=dataset.transform_matrices[7], 
        state=state,
        file_name='instant_rendered_image_2'
    )

if __name__ == '__main__':
    main()