#!/usr/bin/env python3
"""
CSV File Cleaner - Fix concatenated CSV files
Author: TAQDEES
"""

import pandas as pd
import os

def clean_concatenated_csv(input_file, output_file):
    """Clean CSV file with multiple headers"""
    
    print(f"🧹 Cleaning: {input_file}")
    print("=" * 50)
    
    # Read all lines
    with open(input_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Find header lines
    header_lines = []
    for i, line in enumerate(lines):
        if line.strip().startswith('smiles,'):
            header_lines.append(i)
    
    print(f"📊 Total lines: {len(lines)}")
    print(f"📋 Headers found at lines: {[h+1 for h in header_lines]}")
    
    if len(header_lines) <= 1:
        print("✅ No multiple headers detected. File is clean.")
        return False
    
    # Process each section
    all_dataframes = []
    headers_info = []
    
    for i, header_pos in enumerate(header_lines):
        # Determine section boundaries
        if i + 1 < len(header_lines):
            end_pos = header_lines[i + 1]
        else:
            end_pos = len(lines)
        
        section_lines = lines[header_pos:end_pos]
        
        # Get header
        header = section_lines[0].strip().split(',')
        data_lines = len(section_lines) - 1
        
        headers_info.append({
            'section': i + 1,
            'header_line': header_pos + 1,
            'columns': len(header),
            'data_rows': data_lines,
            'header': header
        })
        
        print(f"\n📦 Section {i+1}:")
        print(f"   Lines: {header_pos+1} to {end_pos}")
        print(f"   Columns: {len(header)}")
        print(f"   Data rows: {data_lines}")
        
        # Create temporary DataFrame
        try:
            temp_file = f'temp_section_{i}.csv'
            with open(temp_file, 'w', encoding='utf-8') as f:
                f.writelines(section_lines)
            
            section_df = pd.read_csv(temp_file)
            all_dataframes.append(section_df)
            
            os.remove(temp_file)  # Clean up
            print(f"   ✅ Successfully parsed")
            
        except Exception as e:
            print(f"   ❌ Error parsing: {e}")
    
    if not all_dataframes:
        print("❌ No valid sections found")
        return False
    
    # Find common columns
    print(f"\n🔍 Analyzing column compatibility:")
    all_columns = [set(df.columns) for df in all_dataframes]
    common_columns = all_columns[0]
    
    for i, cols in enumerate(all_columns[1:], 1):
        common_columns = common_columns.intersection(cols)
        print(f"   Section 1 ∩ Section {i+1}: {len(common_columns)} common columns")
    
    common_columns = list(common_columns)
    print(f"\n✅ Final common columns ({len(common_columns)}): {common_columns}")
    
    # Show columns that will be dropped
    for i, df in enumerate(all_dataframes):
        dropped_cols = set(df.columns) - set(common_columns)
        if dropped_cols:
            print(f"⚠️  Section {i+1} dropping columns: {list(dropped_cols)}")
    
    # Combine data
    print(f"\n🔄 Combining sections...")
    combined_dfs = []
    total_rows = 0
    
    for i, df in enumerate(all_dataframes):
        section_data = df[common_columns]
        combined_dfs.append(section_data)
        total_rows += len(section_data)
        print(f"   Section {i+1}: {len(section_data)} rows")
    
    # Final combination
    final_df = pd.concat(combined_dfs, ignore_index=True)
    
    print(f"\n📊 Final result:")
    print(f"   Total rows: {len(final_df)}")
    print(f"   Total columns: {len(final_df.columns)}")
    print(f"   Columns: {list(final_df.columns)}")
    
    # Save cleaned file
    final_df.to_csv(output_file, index=False)
    print(f"\n💾 Cleaned file saved as: {output_file}")
    
    # Create backup of original
    backup_file = input_file.replace('.csv', '_backup.csv')
    if not os.path.exists(backup_file):
        with open(input_file, 'r', encoding='utf-8') as src, open(backup_file, 'w', encoding='utf-8') as dst:
            dst.write(src.read())
        print(f"📦 Original backed up as: {backup_file}")
    
    return True

def main():
    """Main cleaning function"""
    
    print("🧹 CSV File Cleaner")
    print("=" * 50)
    
    input_file = 'processed_data/egfr_type1_filtered.csv'
    output_file = 'processed_data/egfr_type1_filtered_clean.csv'
    
    if not os.path.exists(input_file):
        print(f"❌ File not found: {input_file}")
        return
    
    success = clean_concatenated_csv(input_file, output_file)
    
    if success:
        print("\n🎉 SUCCESS!")
        print("=" * 20)
        print("✅ CSV file cleaned successfully")
        print(f"✅ Clean file: {output_file}")
        print(f"✅ Original backed up")
        print("\n💡 Next steps:")
        print("1. Verify the cleaned file looks correct")
        print("2. Update your training script to use the clean file")
        print("3. Or rename the clean file to replace the original")
    else:
        print("\n❌ No cleaning needed or failed")

if __name__ == "__main__":
    main()