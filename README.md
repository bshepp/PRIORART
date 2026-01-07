# PRIORART

**Systematic Prior Art Generation for Mathematical Method Combinations**

PRIORART is an umbrella framework for exhaustively exploring and documenting combinations of signal processing methods. By systematically testing all parameter combinations and publishing the results with cryptographic checksums, this project establishes public prior art to prevent future patent claims on these techniques.

## Mission

> "Let's do this for all math."

Mathematical methods belong to everyone. This project creates a documented, timestamped, checksummed record of method combinations before patent trolls can claim them.

## Structure

```
PRIORART/
├── core/                      # Domain-agnostic infrastructure
│   ├── decomposition/         # 24+ decomposition methods
│   ├── upsampling/            # 16+ interpolation methods
│   ├── exhaustive.py          # Combination runner
│   ├── checksums.py           # SHA256 generation
│   └── analysis.py            # Redundancy detection
├── domains/
│   ├── scalar2d/              # 2D scalar fields (thermal, pressure, etc.)
│   │   ├── generators.py      # Synthetic test data
│   │   └── methods.py         # Domain extensions
│   └── terrain/               # DEMs (placeholder)
└── experiments/               # Output directories
```

## Decomposition Methods

| Category | Methods |
|----------|---------|
| Classical | gaussian, gaussian_anisotropic, uniform, median |
| Edge-Preserving | bilateral, guided, anisotropic_diffusion |
| Morphological | opening, closing, tophat, rolling_ball, gradient |
| Wavelet | dwt (haar, db2-db8, sym4, coif2, bior, rbio) |
| Multi-scale | dog, dog_multiscale, log |
| Trend Removal | polynomial (deg 1-6), local_polynomial |

## Upsampling Methods

| Category | Methods |
|----------|---------|
| Interpolation | nearest, bilinear, bicubic, quartic, quintic |
| Spline | bspline, cubic_catmull_rom, cubic_mitchell |
| Frequency | fft_zeropad, sinc_hamming, sinc_blackman |
| Adaptive | edge_directed, regularized |
| OpenCV | lanczos, area, linear_exact |

## Usage

### Generate Test Data

```bash
cd domains/scalar2d
python generators.py --output ../../experiments/test_fields --size 256
```

### Run Exhaustive Exploration

```bash
python -m core.exhaustive experiments/test_fields/mixed_features.npy \
    --output experiments/scalar2d_exhaustive \
    --skip-existing
```

### Generate Checksums

```bash
python -m core.checksums experiments/scalar2d_exhaustive/results \
    --output experiments/scalar2d_exhaustive/CHECKSUMS.txt
```

### Analyze Redundancy

```bash
python -m core.analysis \
    --results experiments/scalar2d_exhaustive/results \
    --checksums experiments/scalar2d_exhaustive/CHECKSUMS.txt \
    --output experiments/REDUNDANCY_REPORT.md
```

## Prior Art Strategy

1. **Exhaustive Exploration**: Every parameter combination is tested
2. **SHA256 Checksums**: Cryptographic proof of output content
3. **Git Timestamps**: Commit dates establish priority
4. **Apache 2.0 License**: Explicit patent grant with retaliation clause
5. **Public Publication**: GitHub provides accessible evidence

## Adding New Domains

1. Create `domains/your_domain/`
2. Add `generators.py` with synthetic test data functions
3. Optionally add `methods.py` for domain-specific extensions
4. Run exhaustive exploration on your test data

## Related Projects

- [RESIDUALS](https://github.com/bshepp/RESIDUALS) - Prior art for terrain/LiDAR feature detection
- [terravector](https://github.com/bshepp/terravector) - Terrain patch similarity search

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

