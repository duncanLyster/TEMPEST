# src/tests/test_spice_interface.py

"""
Unit tests for SPICE interface module.

Note: These tests require SPICE kernels to be available.
Some tests may be skipped if kernels are not found.
"""

import unittest
import numpy as np
import os
from pathlib import Path


class TestSpiceInterface(unittest.TestCase):
    """Test cases for SpiceManager class."""
    
    @classmethod
    def setUpClass(cls):
        """Check if test kernels are available."""
        cls.kernel_dir = Path("kernels")
        cls.has_kernels = cls.kernel_dir.exists()
        
        if not cls.has_kernels:
            print("\nWarning: SPICE kernels not found. Some tests will be skipped.")
            print("To run full tests, create a 'kernels/' directory with:")
            print("  - naif0012.tls (leap seconds)")
            print("  - de438.bsp (planetary ephemeris)")
            
    def test_import(self):
        """Test that SpiceManager can be imported."""
        try:
            from src.model.spice_interface import SpiceManager
            self.assertTrue(True)
        except ImportError as e:
            self.fail(f"Failed to import SpiceManager: {e}")
            
    @unittest.skipUnless(os.path.exists("kernels/naif0012.tls"), "Requires leap seconds kernel")
    def test_time_conversion(self):
        """Test UTC to ephemeris time conversion."""
        from src.model.spice_interface import SpiceManager
        
        # Use minimal kernels for time conversion
        kernels = ["kernels/naif0012.tls"]
        
        with SpiceManager(kernels, "EARTH", "SUN") as sm:
            # Test a known time
            time_str = "2000-01-01T12:00:00"
            et = sm.time_str_to_et(time_str)
            
            # J2000 epoch is 2000-01-01T11:58:55.816, so 12:00:00 should be ~64.184 seconds later
            self.assertAlmostEqual(et, 64.184, delta=1.0)
            
            # Test round-trip conversion
            time_str_out = sm.et_to_time_str(et)
            self.assertIn("2000-001", time_str_out)  # Should contain year and day
            
    @unittest.skipUnless(
        os.path.exists("kernels/naif0012.tls") and os.path.exists("kernels/de438.bsp"),
        "Requires leap seconds and planetary ephemeris kernels"
    )
    def test_sun_direction(self):
        """Test getting Sun direction and distance."""
        from src.model.spice_interface import SpiceManager
        
        kernels = [
            "kernels/naif0012.tls",
            "kernels/de438.bsp"
        ]
        
        with SpiceManager(kernels, "EARTH", "SUN") as sm:
            et = sm.time_str_to_et("2024-01-01T00:00:00")
            sun_dir, sun_dist = sm.get_sun_direction_and_distance(et, in_body_frame=False)
            
            # Check that direction is normalized
            self.assertAlmostEqual(np.linalg.norm(sun_dir), 1.0, places=6)
            
            # Check that distance is reasonable (Earth is ~1 AU from Sun)
            au_to_m = 1.496e11
            distance_au = sun_dist / au_to_m
            self.assertGreater(distance_au, 0.98)
            self.assertLess(distance_au, 1.02)
            
    def test_context_manager(self):
        """Test that context manager properly cleans up."""
        from src.model.spice_interface import SpiceManager
        
        # This should not raise an error even with invalid kernels
        # because cleanup should handle missing kernels gracefully
        try:
            with SpiceManager([], "EARTH", "SUN") as sm:
                pass
        except RuntimeError as e:
            # Expected - no kernels loaded
            self.assertIn("No SPICE kernels", str(e))
        
    def test_missing_kernels_warning(self):
        """Test that missing kernel files produce warnings."""
        from src.model.spice_interface import SpiceManager
        import warnings
        
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            try:
                sm = SpiceManager(["nonexistent_kernel.bsp"], "EARTH", "SUN")
                sm.cleanup()
            except RuntimeError:
                # Expected - no valid kernels
                pass
                
            # Check that a warning was issued
            self.assertTrue(any("not found" in str(warning.message) for warning in w))


class TestObserver(unittest.TestCase):
    """Test cases for Observer class."""
    
    def test_import(self):
        """Test that Observer can be imported."""
        try:
            from src.model.observer import Observer
            self.assertTrue(True)
        except ImportError as e:
            self.fail(f"Failed to import Observer: {e}")
            
    def test_manual_observer(self):
        """Test Observer with manual position."""
        from src.model.observer import Observer
        
        position = np.array([1000.0, 0.0, 0.0])  # 1 km along x-axis
        observer = Observer("TestObserver", manual_position=position)
        
        # Test getting position
        obs_pos = observer.get_position()
        np.testing.assert_array_equal(obs_pos, position)
        
    def test_direction_to_observer(self):
        """Test calculating direction to observer."""
        from src.model.observer import Observer
        
        observer_pos = np.array([1000.0, 0.0, 0.0])
        observer = Observer("TestObserver", manual_position=observer_pos)
        
        surface_point = np.array([0.0, 0.0, 0.0])
        direction = observer.get_direction_to_observer(surface_point)
        
        # Direction should be along positive x-axis
        expected = np.array([1.0, 0.0, 0.0])
        np.testing.assert_array_almost_equal(direction, expected)
        
    def test_facet_visibility(self):
        """Test facet visibility calculation."""
        from src.model.observer import Observer
        
        observer_pos = np.array([1000.0, 0.0, 0.0])
        observer = Observer("TestObserver", manual_position=observer_pos)
        
        # Facet facing observer
        facet_pos = np.array([0.0, 0.0, 0.0])
        facet_normal = np.array([1.0, 0.0, 0.0])  # Pointing toward observer
        
        is_visible = observer.is_facet_visible(facet_pos, facet_normal)
        self.assertTrue(is_visible)
        
        # Facet facing away from observer
        facet_normal_away = np.array([-1.0, 0.0, 0.0])  # Pointing away
        is_visible_away = observer.is_facet_visible(facet_pos, facet_normal_away)
        self.assertFalse(is_visible_away)
        
    def test_emission_angle(self):
        """Test emission angle calculation."""
        from src.model.observer import Observer
        
        observer_pos = np.array([1000.0, 0.0, 0.0])
        observer = Observer("TestObserver", manual_position=observer_pos)
        
        facet_pos = np.array([0.0, 0.0, 0.0])
        
        # Facet normal aligned with observer direction -> 0 degrees
        facet_normal = np.array([1.0, 0.0, 0.0])
        emission = observer.calculate_emission_angle(facet_pos, facet_normal)
        self.assertAlmostEqual(emission, 0.0, places=2)
        
        # Facet normal perpendicular to observer direction -> 90 degrees
        facet_normal_perp = np.array([0.0, 1.0, 0.0])
        emission_perp = observer.calculate_emission_angle(facet_pos, facet_normal_perp)
        self.assertAlmostEqual(emission_perp, 90.0, places=2)


class TestSpiceConfiguration(unittest.TestCase):
    """Test SPICE configuration handling."""
    
    def test_config_validation(self):
        """Test SPICE configuration validation."""
        from src.utilities.config import Config
        import tempfile
        import yaml
        
        # Create a temporary config file with SPICE settings
        config_data = {
            'use_spice': True,
            'spice_kernels': ['kernel1.bsp'],
            'spice_target_body': 'EARTH',
            'spice_start_time': '2024-01-01T00:00:00',
            'spice_duration_hours': 24,
            'shape_model_file': 'test.stl',
            'emissivity': 0.9
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            temp_path = f.name
            
        try:
            config = Config(config_path=temp_path)
            self.assertTrue(config.use_spice)
            self.assertEqual(config.spice_target_body, 'EARTH')
            self.assertEqual(config.spice_duration_hours, 24)
        finally:
            os.unlink(temp_path)


def run_tests():
    """Run all tests."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test cases
    suite.addTests(loader.loadTestsFromTestCase(TestSpiceInterface))
    suite.addTests(loader.loadTestsFromTestCase(TestObserver))
    suite.addTests(loader.loadTestsFromTestCase(TestSpiceConfiguration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    exit(0 if success else 1)

