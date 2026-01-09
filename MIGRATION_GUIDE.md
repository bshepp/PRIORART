# PRIORART Migration Guide

This guide explains how to migrate `*vector` projects to use PRIORART as the single source of truth for feature extraction.

## Why Migrate?

Currently, feature extraction code is duplicated across projects:
- terravector, astrovector, cmbvector, satvector, fieldvector

This leads to:
- Bug fixes applied 5+ times
- Code divergence over time
- Maintenance burden

## New Architecture

```
PRIORART (source of truth)
├── core/                    # Infrastructure
│   ├── plugin.py           # Plugin interfaces
│   ├── registry.py         # Plugin discovery
│   ├── analysis.py         # Redundancy analysis
│   └── fingerprinting.py   # Parallel fingerprinting
└── domains/                 # Feature extraction
    ├── terrain/            # DEM processing
    ├── astronomy/          # Galaxy morphology
    ├── cmb/                # CMB statistics
    └── satellite/          # Satellite imagery

terravector (thin app layer)
├── app.py                  # Gradio UI (keep)
├── cli.py                  # Build/query CLI (keep)
└── src/
    ├── embedding.py        # HNSW indexing (keep)
    ├── tiling.py           # Patch extraction (keep)
    └── features/           # DELETE - import from priorart.domains.terrain
```

## Migration Steps for terravector

### Step 1: Add PRIORART dependency

```bash
# In terravector directory
pip install -e ../PRIORART
```

Or in `requirements.txt`:
```
priorart @ file:///../PRIORART
```

### Step 2: Update imports

**Before** (terravector/src/embedding.py):
```python
from .decomposition import run_decomposition
from .upsampling import run_upsampling
from .features.geomorphometric import compute_geomorphometric_features
```

**After**:
```python
from priorart.core.decomposition import run_decomposition
from priorart.core.upsampling import run_upsampling
from priorart.domains.terrain import TerrainPlugin

# Get terrain features from PRIORART
terrain = TerrainPlugin()
```

### Step 3: Delete duplicated code

Remove these directories from terravector:
- `src/decomposition/`
- `src/upsampling/`
- `src/features/` (if all features are now in PRIORART)

### Step 4: Test

```bash
python -m pytest tests/
python cli.py build test_dem.npy --output test.idx
python cli.py query test.idx --patch 100,100
```

## Migration for Other Projects

### astrovector

Replace:
```python
from .features.morphology import compute_concentration, compute_asymmetry
```

With:
```python
from priorart.domains.astronomy import compute_concentration, compute_asymmetry
```

### cmbvector

Replace:
```python
from .features.power_spectrum import compute_power_spectrum
```

With:
```python
from priorart.domains.cmb import compute_power_spectrum
```

### satvector

Replace:
```python
from .features.texture import compute_glcm_features
```

With:
```python
from priorart.domains.satellite import compute_all_texture
```

## Benefits After Migration

1. **Single source of truth**: Bug fixes apply everywhere
2. **Smaller codebases**: Each `*vector` project is ~500 lines instead of 2000+
3. **Consistent APIs**: All projects use the same feature interfaces
4. **Exhaustive testing**: PRIORART's exhaustive runner tests all combinations
5. **Prior art**: Feature combinations are documented and checksummed

## Gradual Migration

You don't have to migrate everything at once:

1. Start by importing individual functions from PRIORART
2. Keep local copies as fallbacks
3. Remove local copies once PRIORART versions are tested
4. Eventually, `*vector` projects become thin UI/indexing layers

## Questions?

Open an issue on the PRIORART repository.
