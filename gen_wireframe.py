import sys
from pathlib import Path
current_dir = Path(__file__).resolve().parent
sys.path.append(str(current_dir))
from src.model.spherical_cap_mesh import visualize_spherical_cap

visualize_spherical_cap(2000, 90.0)
