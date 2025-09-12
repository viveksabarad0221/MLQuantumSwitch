# MLQuantumSwitch

Tools for measuring quantum measurement incompatibility and clustering observables, with utilities to generate synthetic noisy datasets and visualize results. The core logic lives in `Code/compatibility_tools.py`, and an example exploration notebook is `Code/mlqs_compatibility.ipynb`.

## What’s here
- `Code/compatibility_tools.py` — Main library module providing:
  - Constructing qubit observables/POVMs (Pauli, projective along a Bloch direction)
  - Validating POVMs
  - Incompatibility measures via mutual eigenspace disturbance (analytical, numerical, experimental)
  - Sampling Bloch vectors on spherical caps and building POVMs
  - Simple clustering on distance matrices (k-medoids, k-means, optional HDBSCAN)
- `Code/mlqs_compatibility.ipynb` — Notebook that demonstrates dataset generation and clustering/plots
- `Plots&Data/` — Figures and saved datasets (npz/pkl)

## Install
Create a virtual environment and install the dependencies.

```powershell
# PowerShell (Windows)
python -m venv .venv ; .\.venv\Scripts\Activate.ps1
pip install -U pip
pip install -r requirements.txt
```

Optional (only if you want HDBSCAN clustering):
```powershell
pip install hdbscan
```

## Quick start
Minimal example: construct two qubit POVMs and compute their incompatibility.

```python
import numpy as np
from Code.compatibility_tools import CompatibilityMeasure

# X and Z axes
x, z = np.array([1,0,0]), np.array([0,0,1])
Ex = CompatibilityMeasure.projective_qubit_povm_from_axis(x)
Ez = CompatibilityMeasure.projective_qubit_povm_from_axis(z)

# Analytical mutual eigenspace disturbance (MED)
med_xz = CompatibilityMeasure.mutual_eigenspace_disturbance(Ex, Ez, method="analytical")
print("MED(X,Z) =", med_xz)
```

Generate a small noisy dataset and plot clustering results (see the notebook for a richer workflow):

```python
import numpy as np
from Code.compatibility_tools import CompatibilityMeasure, ClusteringToolkit

n = 20
spread_angle = 22.5
etas = [0.25, 0.50]

# Sample Bloch directions in cones around +x and +z, then build projective POVMs
vx = CompatibilityMeasure.sample_unit_vectors_cone(n//2, theta_deg=spread_angle, axis=[1,0,0])
vz = CompatibilityMeasure.sample_unit_vectors_cone(n//2, theta_deg=spread_angle, axis=[0,0,1])
vectors = np.vstack([vx, vz])
povms = CompatibilityMeasure.generate_povms_from_bloch_vectors(vectors)

# Pairwise incompatibility distance matrix
D = CompatibilityMeasure.incompatibility_distance_matrix(povms, method="analytical")

# Cluster using k-medoids
labels = ClusteringToolkit.cluster_from_distance(D, n_clusters=2, method="k-medoids")
print("cluster labels:", labels)
```

## Project structure
```
MLQuantumSwitch/
├─ Code/
│  ├─ compatibility_tools.py
│  └─ mlqs_compatibility.ipynb
├─ Plots&Data/
│  └─ MLQS/  # figures and (optionally) saved datasets
├─ README.md
├─ requirements.txt
├─ LICENSE
├─ CONTRIBUTING.md
├─ .gitignore
└─ .gitattributes
```

## Development tips
- Python >= 3.10 recommended (3.12 works)
- Keep functions stateless and add docstrings and type hints
- Prefer numpy and QuTiP primitives for quantum objects; ensure POVMs are positive and complete

## Initialize Git and push to GitHub
After these files are in place, you can create a repo and push. Replace YOUR-USER and REPO-NAME as needed.

```powershell
# In the MLQuantumSwitch folder
git init
git add .
git commit -m "chore: bootstrap repo with docs and scaffolding"

# Create a new empty repo on GitHub first (via the website), then set it as origin
# Replace below with your actual remote URL
 git remote add origin https://github.com/YOUR-USER/REPO-NAME.git
 git branch -M main
 git push -u origin main
```

If you prefer SSH:
```powershell
 git remote add origin git@github.com:YOUR-USER/REPO-NAME.git
 git branch -M main
 git push -u origin main
```

## License
This project is released under the MIT License — see `LICENSE` for details.
