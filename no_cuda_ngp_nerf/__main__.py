import jax
from no_cuda_ngp_nerf.dataset import load_dataset
from no_cuda_ngp_nerf.training import create_train_state, train_loop
from no_cuda_ngp_nerf.rendering import turntable_render
from no_cuda_ngp_nerf.model import InstantNerf
from functools import partial
import argparse

def main():
    print('GPU:', jax.devices('gpu'))
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str)
    parser.add_argument('--downscale_factor', type=int, default=1)
    parser.add_argument('--num_hash_table_levels', type=int, default=16)
    parser.add_argument('--max_hash_table_entries', type=int, default=2**20)
    parser.add_argument('--hash_table_feature_dim', type=int, default=2)
    parser.add_argument('--coarsest_resolution', type=int, default=16)
    parser.add_argument('--finest_resolution', type=int, default=2**19)
    parser.add_argument('--density_mlp_width', type=int, default=16)
    parser.add_argument('--color_mlp_width', type=int, default=64)
    parser.add_argument('--high_dynamic_range', type=bool, default=False)
    parser.add_argument('--exponential_density_activation', type=bool, default=False)
    parser.add_argument('--learning_rate', type=float, default=1e-2)
    parser.add_argument('--epsilon', type=float, default=1e-15)
    parser.add_argument('--weight_decay_coefficient', type=float, default=1e-6)
    parser.add_argument('--ray_near', type=float, default=0.2)
    parser.add_argument('--ray_far', type=float, default=3.0)
    parser.add_argument('--batch_size', type=float, default=30_000)
    parser.add_argument('--train_target_samples_per_ray', type=int, default=32)
    parser.add_argument('--render_max_samples_per_ray', type=int, default=128)
    parser.add_argument('--training_steps', type=int, default=200)
    parser.add_argument('--turntable_render', type=bool, default=True)
    parser.add_argument('--num_turntable_render_frames', type=int, default=3)
    parser.add_argument('--turntable_camera_distance', type=float, default=1.4)
    parser.add_argument('--render_patch_size', type=int, default=32)
    args = parser.parse_args()

    train_max_rays = args.batch_size // args.train_target_samples_per_ray
    print('Train max rays:', train_max_rays)
    assert args.ray_near < args.ray_far, 'Ray near must be less than ray far.'

    dataset = load_dataset(args.dataset_path, args.downscale_factor)
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
        number_of_grid_levels=args.num_hash_table_levels,
        max_hash_table_entries=args.max_hash_table_entries,
        hash_table_feature_dim=args.hash_table_feature_dim,
        coarsest_resolution=args.coarsest_resolution,
        finest_resolution=args.finest_resolution,
        density_mlp_width=args.density_mlp_width,
        color_mlp_width=args.color_mlp_width,
        high_dynamic_range=args.high_dynamic_range,
        exponential_density_activation=args.exponential_density_activation
    )
    rng = jax.random.PRNGKey(1)
    state = create_train_state(
        model=model, rng=rng, learning_rate=args.learning_rate, 
        epsilon=args.epsilon, weight_decay_coefficient=args.weight_decay_coefficient
    )
    del rng
    state = train_loop(
        batch_size=args.batch_size, max_num_rays=train_max_rays, 
        training_steps=args.training_steps, state=state, dataset=dataset
    )

    if args.turntable_render:
        turntable_render(
            num_frames=args.num_turntable_render_frames,
            max_samples_per_ray=args.render_max_samples_per_ray,
            patch_size_x=args.render_patch_size, 
            patch_size_y=args.render_patch_size, 
            camera_distance=args.turntable_camera_distance, 
            ray_near=args.ray_near, 
            ray_far=args.ray_far, 
            state=state, 
            dataset=dataset, 
            file_name='instant_turntable_render'
        )

if __name__ == '__main__':
    main()