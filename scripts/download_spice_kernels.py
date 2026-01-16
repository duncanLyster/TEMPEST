"""
Script to download SPICE kernels for TEMPEST radiance retrieval testing.

This script downloads the necessary SPICE kernels for OSIRIS-REx/Bennu observations.
"""

import os
import urllib.request
from pathlib import Path

# Base URL for NAIF SPICE data
NAIF_BASE_URL = "https://naif.jpl.nasa.gov/pub/naif/"

# Kernel URLs - Updated paths for OSIRIS-REx PDS4 archive
# Base: https://naif.jpl.nasa.gov/pub/naif/pds/pds4/orex/orex_spice/spice_kernels/
KERNELS = {
    'bennu': {
        # Bennu frame kernel - check fk/ directory
        'bennu_v007.tf': 'pds/pds4/orex/orex_spice/spice_kernels/fk/bennu_v007.tf',
        # Bennu physical constants - check pck/ directory  
        'bennu_nav_v007.tpc': 'pds/pds4/orex/orex_spice/spice_kernels/pck/bennu_nav_v007.tpc',
    },
    'orx': {
        # OSIRIS-REx frame kernel
        'orx_v14.tf': 'pds/pds4/orex/orex_spice/spice_kernels/fk/orx_v14.tf',
        # OSIRIS-REx SPK - check spk/ directory for available files
        'orx_20190901_20200101_v01.bsp': 'pds/pds4/orex/orex_spice/spice_kernels/spk/orx_20190901_20200101_v01.bsp',
        # CK (pointing) kernel covering January 15, 2019 - spacecraft attitude (try SC version)
        'orx_sc_rel_190114_190120_v01.bc': 'pds/pds4/orex/orex_spice/spice_kernels/ck/orx_sc_rel_190114_190120_v01.bc',
        # SCLK (spacecraft clock) kernel - latest version
        'orx_sclkscet_00049.tsc': 'pds/pds4/orex/orex_spice/spice_kernels/sclk/orx_sclkscet_00049.tsc',
        # IK (instrument kernel) for OTES
        'orx_otes_v00.ti': 'pds/pds4/orex/orex_spice/spice_kernels/ik/orx_otes_v00.ti',
    },
    'naif': {
        # Generic leap seconds kernel
        'naif0012.tls': 'generic_kernels/lsk/naif0012.tls',
        # Generic planetary constants
        'pck00010.tpc': 'generic_kernels/pck/pck00010.tpc',
    }
}

def download_kernel(url_path, output_path):
    """Download a SPICE kernel file."""
    url = NAIF_BASE_URL + url_path
    
    try:
        print(f"Downloading {os.path.basename(output_path)}...")
        urllib.request.urlretrieve(url, output_path)
        print(f"  ✓ Downloaded to {output_path}")
        return True
    except Exception as e:
        print(f"  ✗ Failed to download: {e}")
        print(f"  URL: {url}")
        return False

def main():
    """Download all required SPICE kernels."""
    script_dir = Path(__file__).resolve().parent
    project_root = script_dir.parent
    spice_dir = project_root / "data" / "spice"
    
    # Create directories
    for subdir in ['bennu', 'orx', 'naif']:
        (spice_dir / subdir).mkdir(parents=True, exist_ok=True)
    
    print("Downloading SPICE kernels for OSIRIS-REx/Bennu...")
    print("=" * 60)
    
    success_count = 0
    total_count = 0
    
    # Download generic kernels first (these are most reliable)
    print("\nNAIF generic kernels:")
    for filename, url_path in KERNELS['naif'].items():
        total_count += 1
        output_path = spice_dir / 'naif' / filename
        if download_kernel(url_path, output_path):
            success_count += 1
    
    # Try OSIRIS-REx/Bennu kernels (may need manual download)
    print("\nOSIRIS-REx/Bennu kernels:")
    print("Note: Mission-specific kernels may need to be downloaded manually.")
    print("Check: https://naif.jpl.nasa.gov/pub/naif/pds/pds4/orex/orex_spice/spice_kernels/")
    
    # Download mission-specific kernels from KERNELS dict
    for category in ['bennu', 'orx']:
        if category in KERNELS:
            print(f"\n{category.upper()} kernels:")
            for filename, url_path in KERNELS[category].items():
                total_count += 1
                output_path = spice_dir / category / filename
                if download_kernel(url_path, output_path):
                    success_count += 1
                else:
                    print(f"  → Manual download may be needed for {filename}")
                    print(f"    Check the OSIRIS-REx SPICE archive for correct filename")
    
    # SPK files are large and version-specific - provide instructions
    print("\nSPK (trajectory) kernels:")
    print("  → SPK files are large and version-specific.")
    print("  → Download manually from:")
    print("    https://naif.jpl.nasa.gov/pub/naif/pds/pds4/orex/orex_spice/spice_kernels/spk/")
    print("  → Look for files covering your observation time period")
    print("  → Example: orx_20190901_20200101_v01.bsp")
    
    print("\n" + "=" * 60)
    print(f"Downloaded {success_count}/{total_count} kernels")
    
    if success_count < total_count:
        print("\n" + "=" * 60)
        print("MANUAL DOWNLOAD INSTRUCTIONS:")
        print("=" * 60)
        print("\n1. Visit: https://naif.jpl.nasa.gov/pub/naif/pds/pds4/orex/orex_spice/spice_kernels/")
        print("\n2. Download the following kernel types:")
        print("   - fk/  : Frame kernels (bennu_v*.tf, orx_v*.tf)")
        print("   - pck/ : Physical constants (bennu_nav_v*.tpc)")
        print("   - spk/ : Trajectory (orx_*.bsp covering your observation dates)")
        print("\n3. Place files in:")
        print(f"   - Bennu kernels: {spice_dir / 'bennu'}")
        print(f"   - OSIRIS-REx kernels: {spice_dir / 'orx'}")
        print(f"   - Generic kernels: {spice_dir / 'naif'}")
        print("\n4. Update kernel paths in your config file if filenames differ")

if __name__ == "__main__":
    main()
