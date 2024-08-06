import imageio.v2 as imageio  # Use the updated import to avoid deprecation warnings
import os

# Define base directories
base_dir = 'D:\\Utilities\\Programing\\Python REPOS\\GAN'
features_dir = os.path.join(base_dir, 'plots', 'features')
iterations_dir = os.path.join(base_dir, 'plots', 'iterations')
images_dir = os.path.join(base_dir, 'images')

# Ensure output directory exists
os.makedirs(images_dir, exist_ok=True)


def create_gif(input_dir, file_pattern, output_file, fps=2):
    images = []
    for i in range(11):
        file_path = os.path.join(input_dir, file_pattern % (i * 1000))
        if os.path.isfile(file_path):
            images.append(imageio.imread(file_path))
        else:
            print(f"Warning: File {file_path} not found. Skipping.")
    if images:
        imageio.mimsave(os.path.join(images_dir, output_file), images, fps=fps)
    else:
        print(f"No images to save for {output_file}")


# Create GIFs
create_gif(features_dir, 'feature_transform_%d.png', 'feature_transform.gif')
create_gif(features_dir, 'feature_transform_centroid_%d.png', 'feature_transform_centroid.gif')
create_gif(iterations_dir, 'iteration_%d.png', 'iterations.gif')
