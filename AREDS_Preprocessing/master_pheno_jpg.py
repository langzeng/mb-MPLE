import pandas as pd
import numpy as np

# 1. Read the image names from the text file
with open('image_name.txt', 'r') as f:
    # Filter out empty lines just in case
    image_names = [line.strip() for line in f if line.strip()]

# 2. Parse filenames to create the base DataFrame
# Format: 2419_24_RE_F2_LS.jpg -> ID2=2419, VISNO=24, EYE=RE, ANGLE=F2, SIDE=LS
parsed_data = []
for img in image_names:
    parts = img.replace('.jpg', '').split('_')
    if len(parts) >= 5:
        parsed_data.append({
            'img_name': img,
            'ID2': parts[0],
            'VISNO': parts[1],
            'EYE': parts[2],
            'ANGLE': parts[3],
            'SIDE': parts[4]
        })

df_img = pd.DataFrame(parsed_data)

# 3. Load the external datasets
# We force ID2 and VISNO to string to ensure they match the parsed image data
pheno_c2 = pd.read_csv(
    'phs000001.v3.pht000375.v2.p1.c2.fundus.GRU.txt',
    skiprows=12,
    sep='\t',
    dtype={'ID2': str, 'VISNO': str}
)

enrollment_c2 = pd.read_csv(
    'phs000001.v3.pht000373.v2.p1.c2.enrollment_randomization.GRU.txt',
    skiprows=10,
    sep='\t',
    dtype={'ID2': str}
)

# 4. Merge Image List with Phenotype Data (on ID2 and VISNO)
# We use a left merge to keep all images, even if phenotype data is missing
df_merged = pd.merge(df_img, pheno_c2, on=['ID2', 'VISNO'], how='left')

# 5. Extract Eye-Specific Columns
# These columns exist as RE... and LE... in pheno_c2. We want a single column.
eye_vars = [
    'DRUSF2', 'NDRUF2', 'SSRF2', 'HDEXF2', 'SUBHF2', 'SUBFF2', 'PHCOF2',
    'GEOACT', 'GEOACS', 'GEOAWI', 'RPEDCI', 'RPEDWI', 'INCPCI', 'INCPWI',
    'DRSZWI', 'DRSOFT', 'DRARWI', 'DRRETI', 'DRCALC', 'ELICAT', 'PHTIME'
]

# Helper logic: if EYE is 'RE', take 'RE'+var, else take 'LE'+var
for var in eye_vars:
    re_col = 'RE' + var
    le_col = 'LE' + var

    # Initialize column
    df_merged[var] = np.nan

    # Fill for Right Eye
    if re_col in df_merged.columns:
        mask_re = df_merged['EYE'] == 'RE'
        df_merged.loc[mask_re, var] = df_merged.loc[mask_re, re_col]

    # Fill for Left Eye
    if le_col in df_merged.columns:
        mask_le = df_merged['EYE'] == 'LE'
        df_merged.loc[mask_le, var] = df_merged.loc[mask_le, le_col]

# Handle AMDSEV specifically (it follows a slightly different naming convention or just to be explicit)
df_merged['AMDSEV'] = np.nan
if 'AMDSEVRE' in df_merged.columns:
    df_merged.loc[df_merged['EYE'] == 'RE', 'AMDSEV'] = df_merged.loc[df_merged['EYE'] == 'RE', 'AMDSEVRE']
if 'AMDSEVLE' in df_merged.columns:
    df_merged.loc[df_merged['EYE'] == 'LE', 'AMDSEV'] = df_merged.loc[df_merged['EYE'] == 'LE', 'AMDSEVLE']

# 6. Merge with Enrollment Data (on ID2)
# Enrollment data is subject-level (not eye/visit specific usually, but check if keys align)
df_final = pd.merge(df_merged, enrollment_c2, on='ID2', how='left')

# 7. Create missing columns requested in the output format but not in files
missing_cols = ['rc_id', 'BATCH', 'SampleID', 'Geno_av']
for col in missing_cols:
    df_final[col] = np.nan

# 8. Select and Order Columns as requested
final_columns = [
    'img_name', 'ID2', 'rc_id', 'VISNO', 'EYE', 'ANGLE', 'SIDE', 'BATCH', 'SampleID', 'Geno_av',
    'AMDSEV', 'DRSZWI', 'ELICAT', 'DRUSF2', 'NDRUF2', 'SSRF2', 'HDEXF2', 'SUBHF2', 'SUBFF2', 'PHCOF2',
    'GEOACT', 'GEOACS', 'GEOAWI', 'RPEDCI', 'RPEDWI', 'INCPCI', 'INCPWI', 'DRSOFT', 'DRARWI', 'DRRETI',
    'DRCALC', 'PHTIME', 'AMDCAT', 'SEX', 'ENROLLAGE', 'MARITAL', 'SCHOOL', 'OCCUPAT', 'RACE',
    'SMOKEDYN', 'SMKAGEST', 'SMKPACKS', 'SMKCURR', 'SMKNOCIG', 'SMKAGEQT', 'BPHIGHYN', 'BPMEDNOW'
]

# Ensure all columns exist (fill with NaN if any from the list are strictly missing from merges)
for col in final_columns:
    if col not in df_final.columns:
        df_final[col] = np.nan

df_final = df_final[final_columns]

# 9. Save to CSV
# Using tab separator as requested in the example format
df_final.to_csv('master_pheno_jpg.txt', sep='\t', index=False)

print("Processing complete. Saved to 'master_pheno_jpg.txt'.")
print(df_final.head())