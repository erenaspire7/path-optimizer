from generate_paths import generate_bezier_curve
from utils import smoothen_path, generate_magnets, sample_along_path

from differential_model import DifferentialModel

if __name__ == "__main__":
    equidistant_points = generate_bezier_curve()

    target_path = smoothen_path(equidistant_points, 100)
    magnet_attributes = generate_magnets(target_path)

    model = DifferentialModel(target_path, magnet_attributes)
    model.run()
