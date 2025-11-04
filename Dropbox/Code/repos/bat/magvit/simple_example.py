#!/usr/bin/env python3
"""
Simple MAGVIT example - validates code structure and basic imports
This is the simplest example that checks MAGVIT setup without requiring full dependencies
"""

import sys
import os

# Add videogvt to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'videogvt'))

print("=" * 70)
print("MAGVIT Simple Example - Code Structure Validation")
print("=" * 70)

errors = []
warnings = []

# Test 1: Check Python version
print("\n📋 Test 1: Python Version")
print(f"   Python: {sys.version.split()[0]}")
if sys.version_info < (3, 8):
    errors.append("Python 3.8+ required")
else:
    print("   ✅ Python version OK")

# Test 2: Check MAGVIT directory structure
print("\n📋 Test 2: Directory Structure")
required_dirs = ['videogvt', 'videogvt/configs', 'videogvt/models', 'videogvt/trainers']
for dir_path in required_dirs:
    full_path = os.path.join(os.path.dirname(__file__), dir_path)
    if os.path.exists(full_path):
        print(f"   ✅ {dir_path}/ exists")
    else:
        errors.append(f"Directory missing: {dir_path}")

# Test 3: Check config files
print("\n📋 Test 3: Configuration Files")
config_files = [
    'videogvt/configs/maskgvt_bair_config.py',
    'videogvt/configs/maskgvt_ucf101_config.py',
    'videogvt/configs/vqgan3d_bair_config.py',
]
for config_file in config_files:
    full_path = os.path.join(os.path.dirname(__file__), config_file)
    if os.path.exists(full_path):
        print(f"   ✅ {config_file} exists")
    else:
        warnings.append(f"Config file missing: {config_file}")

# Test 4: Check main.py
print("\n📋 Test 4: Main Entry Point")
main_py = os.path.join(os.path.dirname(__file__), 'videogvt/main.py')
if os.path.exists(main_py):
    print("   ✅ videogvt/main.py exists")
    with open(main_py, 'r') as f:
        content = f.read()
        if 'def main' in content:
            print("   ✅ main() function found")
        else:
            warnings.append("main() function not found in main.py")
else:
    errors.append("videogvt/main.py missing")

# Test 5: Try basic imports (may fail if dependencies not installed)
print("\n📋 Test 5: Basic Imports")
try:
    import ml_collections
    print("   ✅ ml_collections imported")
except ImportError:
    warnings.append("ml_collections not installed (pip install ml-collections)")

try:
    import jax
    print(f"   ✅ JAX imported (version: {jax.__version__})")
except ImportError:
    warnings.append("JAX not installed (required for MAGVIT)")
except Exception as e:
    warnings.append(f"JAX import issue: {e}")

try:
    # Try importing config (may fail if dependencies missing)
    from videogvt.configs import maskgvt_ucf101_config
    print("   ✅ maskgvt_ucf101_config imported")
except ImportError as e:
    warnings.append(f"Config import failed (may need dependencies): {e}")
except Exception as e:
    warnings.append(f"Config import error: {e}")

# Test 6: Check README
print("\n📋 Test 6: Documentation")
readme = os.path.join(os.path.dirname(__file__), 'README.md')
if os.path.exists(readme):
    print("   ✅ README.md exists")
    with open(readme, 'r') as f:
        content = f.read()
        if 'MAGVIT' in content:
            print("   ✅ README contains MAGVIT information")
else:
    warnings.append("README.md missing")

# Summary
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

if errors:
    print(f"\n❌ Errors ({len(errors)}):")
    for error in errors:
        print(f"   - {error}")
    print("\n⚠️  MAGVIT structure has critical issues. Please fix errors above.")
    sys.exit(1)

if warnings:
    print(f"\n⚠️  Warnings ({len(warnings)}):")
    for warning in warnings:
        print(f"   - {warning}")
    print("\n💡 MAGVIT code structure is OK, but some dependencies may be missing.")
    print("   To run full MAGVIT examples, install dependencies:")
    print("   - Install conda environment: conda env create -f environment.yaml")
    print("   - Or install via pip: pip install -r requirements.txt")
    print("   - Note: CUDA 11 and CuDNN 8.6 required for JAX GPU support")
    sys.exit(0)
else:
    print("\n✅ MAGVIT code structure validated successfully!")
    print("   All basic checks passed.")
    print("\n💡 To run full MAGVIT training/inference:")
    print("   - Install dependencies: conda env create -f environment.yaml")
    print("   - Or: pip install -r requirements.txt")
    print("   - Activate environment: conda activate magvit")
    print("   - Run: python3 videogvt/main.py --config=...")
    sys.exit(0)
