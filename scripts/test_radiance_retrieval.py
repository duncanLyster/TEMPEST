"""
Test script for radiance retrieval functionality.

This script tests the radiance retrieval system without requiring full SPICE kernels
or TEMPEST output files.
"""

import sys
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

def test_planck_radiance():
    """Test Planck radiance calculation."""
    print("Testing Planck radiance calculation...")
    from retrieve_radiance import planck_radiance
    
    wavelengths = np.linspace(5e-6, 50e-6, 10)  # 5-50 microns
    temperature = 300.0  # K
    
    radiance = planck_radiance(wavelengths, temperature, emissivity=0.9)
    
    assert radiance.shape == wavelengths.shape, "Radiance shape mismatch"
    assert np.all(radiance > 0), "Radiance should be positive"
    assert np.all(np.isfinite(radiance)), "Radiance should be finite"
    
    print(f"  ✓ Planck radiance: shape={radiance.shape}, range=[{radiance.min():.2e}, {radiance.max():.2e}] W/(m^2 sr m)")
    return True

def test_fov_rays():
    """Test FOV ray generation."""
    print("Testing FOV ray generation...")
    from retrieve_radiance import generate_fov_rays
    
    observer_pos = np.array([1000.0, 0.0, 0.0])  # 1 km away
    pointing = np.array([-1.0, 0.0, 0.0])  # Pointing toward origin
    fov_angle = np.radians(7.0)  # 7 degree FOV
    
    rays = generate_fov_rays(observer_pos, pointing, fov_angle, n_rays=100)
    
    assert rays.shape == (100, 3), f"Rays shape mismatch: {rays.shape}"
    # Check that rays are normalized
    norms = np.linalg.norm(rays, axis=1)
    assert np.allclose(norms, 1.0, rtol=1e-6), "Rays should be normalized"
    
    # Check that rays are within FOV
    cosines = np.dot(rays, pointing)
    max_angle = np.max(np.arccos(cosines))
    assert max_angle <= fov_angle + 0.1, f"Rays outside FOV: max_angle={np.degrees(max_angle):.2f} deg"
    
    print(f"  ✓ Generated {len(rays)} rays, max angle: {np.degrees(max_angle):.2f} deg")
    return True

def test_spice_import():
    """Test SPICE import."""
    print("Testing SPICE import...")
    try:
        import spiceypy as spice
        print(f"  ✓ spiceypy imported successfully (version: {spice.tkvrsn('toolkit')})")
        return True
    except ImportError as e:
        print(f"  ✗ Failed to import spiceypy: {e}")
        print("  Install with: pip install spicepy")
        return False

def test_config_loading():
    """Test config file loading."""
    print("Testing config file loading...")
    from src.utilities.config import Config
    
    config_path = project_root / "data" / "config" / "radiance_retrievals" / "Bennu_OTES.yaml"
    
    if not config_path.exists():
        print(f"  ✗ Config file not found: {config_path}")
        return False
    
    try:
        config = Config(config_path=str(config_path))
        print(f"  ✓ Config loaded successfully")
        print(f"    Target: {config.target_id}, Observer: {config.observer_id}")
        return True
    except Exception as e:
        print(f"  ✗ Failed to load config: {e}")
        return False

def main():
    """Run all tests."""
    print("=" * 60)
    print("Testing Radiance Retrieval System")
    print("=" * 60)
    
    tests = [
        ("Planck Radiance", test_planck_radiance),
        ("FOV Ray Generation", test_fov_rays),
        ("SPICE Import", test_spice_import),
        ("Config Loading", test_config_loading),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"  ✗ Test failed with exception: {e}")
            results.append((name, False))
        print()
    
    print("=" * 60)
    print("Test Results:")
    print("=" * 60)
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {status}: {name}")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    print(f"\nPassed: {passed}/{total}")
    
    if passed == total:
        print("\n✓ All tests passed!")
        return 0
    else:
        print("\n✗ Some tests failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())
