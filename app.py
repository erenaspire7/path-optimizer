from generate_paths import (
    generate_bezier_curve,
    generate_random_path,
    visualize_smooth_path,
)
from utils import smoothen_path, generate_magnets, sample_along_path

from differential_model import DifferentialModel

if __name__ == "__main__":
    target_path = generate_random_path(5)
    target_path = visualize_smooth_path(target_path, 100)

    # target_path = generate_smooth_random_path()
    magnet_attributes = generate_magnets(target_path)

    model = DifferentialModel(target_path, magnet_attributes)
    model.run()
