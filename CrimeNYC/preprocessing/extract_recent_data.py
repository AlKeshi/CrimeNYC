"""
Pre-processing script to extract recent crime data from the large NYPD CSV file.
This creates a smaller, more manageable CSV file containing only 2020+ data.
Run this first before running the main process_data notebook.
"""

import pandas as pd
from datetime import datetime

print('=' * 60)
print('NYPD Data Extraction Tool')
print('Creating a smaller dataset with recent data (2020+)')
print('=' * 60)

# Configuration
input_file = '../../NYPD_data.csv'
output_file = '../data/NYPD_data_2020plus.csv'
start_year = 2020
chunk_size = 10000  # Very small chunks to avoid memory issues

# Columns we need
cols_to_use = [
    'CMPLNT_NUM', 'CMPLNT_FR_DT', 'OFNS_DESC',
    'X_COORD_CD', 'Y_COORD_CD', 'Latitude', 'Longitude'
]

print(f'\nReading from: {input_file}')
print(f'Writing to: {output_file}')
print(f'Filtering to years >= {start_year}')
print(f'Chunk size: {chunk_size:,} rows\n')

# Process in chunks and write directly to new CSV
first_chunk = True
total_processed = 0
total_kept = 0

try:
    for i, chunk in enumerate(pd.read_csv(
        input_file,
        usecols=cols_to_use,
        chunksize=chunk_size,
        low_memory=False
    )):
        total_processed += len(chunk)

        # Parse dates and filter
        chunk['CMPLNT_FR_DT'] = pd.to_datetime(
            chunk['CMPLNT_FR_DT'],
            format='%m/%d/%Y',
            errors='coerce'
        )

        # Keep only valid dates from start_year onwards
        chunk = chunk.dropna(subset=['CMPLNT_FR_DT'])
        chunk['Year'] = chunk['CMPLNT_FR_DT'].dt.year
        chunk = chunk[chunk['Year'] >= start_year]

        # Clean coordinates
        chunk = chunk.dropna(subset=['X_COORD_CD', 'Y_COORD_CD', 'Latitude', 'Longitude'])
        chunk = chunk[
            (chunk['X_COORD_CD'] != 0) &
            (chunk['Y_COORD_CD'] != 0) &
            (chunk['Latitude'] != 0) &
            (chunk['Longitude'] != 0)
        ]

        # Remove the Year column before saving (we'll recalculate it later)
        chunk = chunk.drop('Year', axis=1)

        # Write to CSV
        if len(chunk) > 0:
            total_kept += len(chunk)
            chunk.to_csv(
                output_file,
                mode='w' if first_chunk else 'a',
                header=first_chunk,
                index=False
            )
            first_chunk = False

        # Progress update every 50 chunks
        if (i + 1) % 50 == 0:
            progress_pct = (total_kept / max(total_processed, 1)) * 100
            print(f'Chunk {i+1:4d}: Processed {total_processed:8,} | '
                  f'Kept {total_kept:7,} ({progress_pct:.1f}%)')

    print('\n' + '=' * 60)
    print('EXTRACTION COMPLETE!')
    print('=' * 60)
    print(f'Total rows processed: {total_processed:,}')
    print(f'Total rows kept: {total_kept:,}')
    print(f'Reduction: {100 - (total_kept/max(total_processed,1)*100):.1f}%')
    print(f'\nNew file created: {output_file}')
    print(f'You can now load this file in the process_data notebook!')

except Exception as e:
    print(f'\nError occurred: {str(e)}')
    print('Try reducing the chunk_size if you see memory errors.')
