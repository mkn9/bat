# MAGVIT Dependency Status

## Current Issue

The Google MAGVIT implementation (in `/bat/magvit/`) requires specific versions of JAX, TensorFlow, Flax, and related packages that have complex interdependencies. Using pip for installation has proven challenging due to dependency resolution conflicts.

## Recommended Solution

**Use conda environment as recommended in MAGVIT README:**

```bash
cd /root/bat/magvit
conda env create -f environment.yaml
conda activate magvit
```

This ensures all packages are installed with compatible versions.

## Alternative: Compatible Versions (Work in Progress)

If conda is not available, the following versions are known to be compatible:

- **numpy**: 1.24.3
- **ml-dtypes**: 0.2.0
- **JAX**: 0.4.23 (with CUDA 11)
- **jaxlib**: 0.4.23
- **TensorFlow**: 2.13.0
- **tensorboard**: 2.13.0
- **Flax**: 0.5.0 (needs verification - API compatibility issues with JAX 0.4.23)
- **optax**: 0.1.7
- **orbax-checkpoint**: 0.3.2
- **typing-extensions**: 4.5.0

**Current blocker**: Flax 0.5.0 expects `ShapedArray` to be directly importable from `jax`, but in JAX 0.4.23 it's in `jax.core.ShapedArray`. This requires either:
1. Using an older Flax version (0.4.x) that works with JAX 0.4.23
2. Using a newer JAX version (0.4.28+) that exports ShapedArray directly
3. Using conda environment as recommended

## Note on Working MAGVIT Project

The working MAGVIT project at `/Users/mike/Dropbox/Code/repos/MAGVIT` uses **PyTorch**, not JAX/TensorFlow. It's a different implementation that avoids these dependency conflicts entirely.

## Status

- ✅ Training script created: `magvit/train_example.py`
- ✅ Main integration: `main_macbook.py` updated with `--run-magvit-training`
- ⚠️  Dependencies: Version conflicts prevent full execution
- 💡 Recommendation: Use conda environment for reliable dependency management

