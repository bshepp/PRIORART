# PRIORART

**Systematic Prior Art Generation for Mathematical Method Combinations**

PRIORART is the **single source of truth** for feature extraction across all `*vector` projects. By hosting all decomposition, upsampling, and domain-specific methods in one place, we eliminate code duplication and ensure consistency.

## Mission

> "Let's do this for all math."

Mathematical methods belong to everyone. This project creates a documented, timestamped, checksummed record of method combinations before patent trolls can claim them.

## Architecture

```
PRIORART (Single Source of Truth)
├── core/                           # Domain-agnostic infrastructure
│   ├── plugin.py                  # Plugin interfaces
│   ├── registry.py                # Plugin discovery
│   ├── exhaustive.py              # Combination runner
│   ├── analysis.py                # Redundancy detection
│   ├── fingerprinting.py          # Parallel fingerprinting
│   ├── checksums.py               # SHA256 generation
│   └── cli.py                     # Unified CLI
├── domains/                        # Feature extraction by domain
│   ├── terrain/                   # DEM decomposition + upsampling
│   ├── astronomy/                 # Galaxy morphology (CAS, Gini-M20)
│   ├── cmb/                       # CMB power spectrum + statistics
│   ├── satellite/                 # Texture + spectral indices
│   └── scalar2d/                  # Generic 2D scalar fields
└── experiments/                    # Output directories

*vector projects (thin app layers)
├── terravector                    # Terrain similarity search
├── astrovector                    # Galaxy classification
├── cmbvector                      # CMB analysis
├── satvector                      # Satellite imagery
└── fieldvector                    # Generic field analysis
```

## Domains

| Domain | Methods | Use Case |
|--------|---------|----------|
| **terrain** | 24 decomposition + 16 upsampling | LiDAR DEMs, terrain features |
| **astronomy** | CAS, Gini-M20, radial profiles | Galaxy morphology |
| **cmb** | Power spectrum, Minkowski | Cosmic Microwave Background |
| **satellite** | GLCM texture, spectral indices | Remote sensing |
| **scalar2d** | Generic statistical features | Any 2D scalar field |

## Quick Start

### Installation

```bash
git clone https://github.com/bshepp/PRIORART
cd PRIORART
pip install -e .
```

### List Available Domains

```bash
python -m priorart domains

# Output:
# PRIORART Domain Registry
# ========================
# Domain: terrain
#   Pipelines: decomp_upsample
#     - decomp_upsample: 39,731 combinations
# ...
```

### Show Domain Details

```bash
python -m priorart info terrain -v
```

### Run Exhaustive Exploration

```bash
# Generate test data
python -c "import numpy as np; np.save('test.npy', np.random.randn(256, 256))"

# Run all combinations
python -m priorart run test.npy -d terrain -o results/

# Or just documentation
python -m priorart run test.npy -d terrain -o results/ --doc-only
```

### Generate Checksums

```bash
python -m priorart checksums results/results/
```

### Run Redundancy Analysis

```bash
python -m priorart analyze results/results/ \
    --checksums results/results/CHECKSUMS.txt \
    -o results/REDUNDANCY_REPORT.md
```

### Parallel Fingerprinting

```bash
python -m priorart fingerprint results/results/ \
    --workers 4 \
    --checkpoint results/fingerprints.json
```

## Importing from *vector Projects

After migration, `*vector` projects import from PRIORART:

```python
# In terravector
from priorart.core.decomposition import run_decomposition
from priorart.core.upsampling import run_upsampling
from priorart.domains.terrain import TerrainPlugin

# In astrovector
from priorart.domains.astronomy import (
    compute_concentration,
    compute_asymmetry,
    compute_gini,
)
```

See [MIGRATION_GUIDE.md](MIGRATION_GUIDE.md) for detailed instructions.

## Plugin System

Create your own domain plugin:

```python
from priorart.core import DomainPlugin, Pipeline, MethodCategory, Method

class MyDomainPlugin(DomainPlugin):
    @property
    def name(self) -> str:
        return "mydomain"
    
    @property
    def description(self) -> str:
        return "My custom domain"
    
    def get_pipelines(self):
        category = MethodCategory(name='features')
        category.register(Method(
            name='my_feature',
            func=compute_my_feature,
            category='features',
            param_ranges={'sigma': [1, 2, 5, 10]}
        ))
        return [Pipeline(name='main', stages=[category])]

# Register for auto-discovery
PLUGIN = MyDomainPlugin()
```

## Prior Art Strategy

1. **Exhaustive Exploration**: Every parameter combination is tested
2. **SHA256 Checksums**: Cryptographic proof of output content
3. **Git Timestamps**: Commit dates establish priority
4. **Apache 2.0 License**: Explicit patent grant with retaliation clause
5. **Public Publication**: GitHub provides accessible evidence

## Proven at Scale

PRIORART's infrastructure has been tested on:
- **39,731 combinations** in the terrain domain
- **4.28 TB** of generated outputs
- **20 distinct clusters** identified via redundancy analysis

See [DIVERGE/PRIOR_ART.md](https://github.com/bshepp/DIVERGE) for the full terrain prior art.

## Related Projects

- [DIVERGE](https://github.com/bshepp/DIVERGE) - Prior art for terrain/LiDAR (4.28 TB)
- [terravector](https://github.com/bshepp/terravector) - Terrain patch similarity search
- [astrovector](https://github.com/bshepp/astrovector) - Galaxy morphology analysis
- [cmbvector](https://github.com/bshepp/cmbvector) - CMB analysis
- [satvector](https://github.com/bshepp/satvector) - Satellite imagery analysis

## License

Apache 2.0 - See [LICENSE](LICENSE)

The Apache 2.0 license provides:
- Explicit patent grant (Section 3)
- Patent retaliation clause (if you sue, you lose the license)
- Clear attribution requirements

## Citation

```bibtex
@software{priorart2026,
  author = {PRIORART Contributors},
  title = {PRIORART: Prior Art Generation Framework},
  year = {2026},
  url = {https://github.com/bshepp/PRIORART}
}
```

## Philosophy

Every combination of mathematical methods documented here is:
- **Public**: Available to anyone
- **Timestamped**: Priority established by git history
- **Verifiable**: Checksums prove content
- **Unpatentable**: Prior art defeats novelty claims

This is a minefield for patent trolls. Step carefully.

---

*Built with NumPy, SciPy, scikit-image, PyWavelets, and OpenCV.*
