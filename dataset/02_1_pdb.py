#!/usr/bin/env python3
"""
Simple PDB EGFR Structure Download
Fast download of EGFR crystal structure for binding site analysis.

This should take 10-30 seconds maximum!
"""

import requests
import time
from pathlib import Path

def download_egfr_structure(output_dir="C:\\Users\\admin\\BF-final-version\\raw_data"):
    """
    Download EGFR crystal structure from PDB.
    Fast and simple - no complexity needed.
    """
    
    print("🚀 Simple EGFR Structure Download from PDB")
    print("="*50)
    print("🏗️ Downloading 1M17: EGFR kinase with erlotinib")
    print("⚡ This should take 10-30 seconds!")
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # PDB structure details
    pdb_id = "1M17"
    description = "EGFR kinase domain with erlotinib (Type 1 inhibitor)"
    resolution = "2.6 Å"
    
    print(f"\n📥 Downloading {pdb_id}...")
    print(f"📋 Description: {description}")
    print(f"🔬 Resolution: {resolution}")
    
    # Download from PDB
    pdb_url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    
    try:
        start_time = time.time()
        
        response = requests.get(pdb_url, timeout=30)
        response.raise_for_status()
        
        download_time = time.time() - start_time
        
        # Save PDB file
        output_file = Path(output_dir) / f"{pdb_id}.pdb"
        with open(output_file, 'w') as f:
            f.write(response.text)
        
        # Get file size
        file_size = output_file.stat().st_size
        file_size_mb = file_size / (1024 * 1024)
        
        print(f"✅ Downloaded successfully!")
        print(f"📁 Saved to: {output_file}")
        print(f"📊 File size: {file_size_mb:.2f} MB")
        print(f"⏱️ Download time: {download_time:.2f} seconds")
        
        # Validate file content
        print(f"\n🔍 Validating PDB file...")
        
        with open(output_file, 'r') as f:
            content = f.read()
            
        # Basic validation checks
        if 'HEADER' in content:
            print("✅ Valid PDB header found")
        else:
            print("⚠️ No PDB header found")
            
        if 'ATOM' in content:
            atom_lines = content.count('ATOM')
            print(f"✅ Found {atom_lines:,} ATOM records")
        else:
            print("❌ No ATOM records found")
            
        if 'HETATM' in content:
            hetatm_lines = content.count('HETATM')
            print(f"✅ Found {hetatm_lines:,} HETATM records (includes ligand)")
        else:
            print("⚠️ No HETATM records found")
            
        # Check for erlotinib ligand (AQ4)
        if 'AQ4' in content:
            print("✅ Erlotinib ligand (AQ4) found in structure")
        else:
            print("⚠️ Erlotinib ligand not found")
            
        # Extract basic info
        lines = content.split('\n')
        header_line = next((line for line in lines if line.startswith('HEADER')), None)
        if header_line:
            print(f"📋 Header: {header_line[10:50].strip()}")
            
        resolution_line = next((line for line in lines if 'RESOLUTION' in line and 'ANGSTROM' in line), None)
        if resolution_line:
            print(f"🔬 Resolution info: {resolution_line.strip()}")
        
        print("\n" + "="*60)
        print("🎉 PDB Structure Download Complete!")
        print("="*60)
        print(f"📁 File: {output_file}")
        print(f"📊 Size: {file_size_mb:.2f} MB")
        print(f"⚡ Time: {download_time:.2f} seconds")
        print(f"🎯 Ready for binding site extraction!")
        print("="*60)
        
        return True
        
    except requests.RequestException as e:
        print(f"❌ Network error downloading {pdb_id}: {e}")
        return False
        
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def download_additional_egfr_structures(output_dir="C:\\Users\\admin\\BF-final-version\\raw_data"):
    """
    Optionally download additional EGFR structures for comparison.
    Only if needed for advanced analysis.
    """
    
    print(f"\n🔄 Optional: Download additional EGFR structures?")
    user_input = input("Download additional structures? (y/n): ").lower().strip()
    
    if user_input not in ['y', 'yes']:
        print("⏭️ Skipping additional structures")
        return True
    
    additional_structures = [
        ("2ITY", "EGFR with gefitinib", "2.7 Å"),
        ("3W2S", "EGFR with lapatinib", "3.0 Å"),
        ("4ZAU", "EGFR T790M with osimertinib", "2.8 Å")
    ]
    
    print(f"\n📥 Downloading {len(additional_structures)} additional structures...")
    
    success_count = 0
    
    for pdb_id, description, resolution in additional_structures:
        print(f"\n📥 Downloading {pdb_id}: {description} ({resolution})")
        
        pdb_url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
        
        try:
            response = requests.get(pdb_url, timeout=30)
            response.raise_for_status()
            
            output_file = Path(output_dir) / f"{pdb_id}.pdb"
            with open(output_file, 'w') as f:
                f.write(response.text)
                
            file_size = output_file.stat().st_size / (1024 * 1024)
            print(f"✅ {pdb_id} downloaded ({file_size:.2f} MB)")
            success_count += 1
            
            time.sleep(1)  # Be nice to PDB servers
            
        except Exception as e:
            print(f"❌ Failed to download {pdb_id}: {e}")
    
    print(f"\n✅ Successfully downloaded {success_count}/{len(additional_structures)} additional structures")
    return success_count > 0

if __name__ == "__main__":
    output_dir = "C:\\Users\\admin\\BF-final-version\\raw_data"
    
    print("🚀 Starting Simple PDB Structure Download")
    print("⚡ This should be super fast!")
    
    total_start = time.time()
    
    # Download main EGFR structure
    success = download_egfr_structure(output_dir)
    
    if success:
        # Optionally download additional structures
        download_additional_egfr_structures(output_dir)
        
        total_time = time.time() - total_start
        
        print(f"\n🎉 All downloads complete in {total_time:.1f} seconds!")
        print("📁 PDB files ready for binding site extraction")
        print("🚀 Next: Run the Type 1 inhibitor processing script")
        
    else:
        print(f"\n❌ Download failed!")
        exit(1)