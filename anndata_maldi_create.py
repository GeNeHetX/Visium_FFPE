import anndata
import pandas as pd
import numpy as np
import tifffile as tiff
from scipy.sparse import csr_matrix

import warnings
warnings.filterwarnings("ignore")

# Hyperparameters
path = "data_hdd"
slide = "13AG00043-26"
downsample = 50

# load the data
peaks = pd.read_feather(f"{path}/{slide}/results/mse_peaks_feather/peaks.feather")
pixels = pd.read_feather(f"{path}/{slide}/results/mse_peaks_feather/pixels.feather")
features = pd.read_feather(f"{path}/{slide}/results/mse_peaks_feather/features.feather")

# Reindex the features to mz values
features.index = features.mz.values

# Chnage the data types from float64 to float32
peaks = peaks.astype(np.float32)

for col in features.columns:
    if features[col].dtype == np.float64:
        features[col] = features[col].astype(np.float32)

for col in pixels.columns:
    if pixels[col].dtype == np.float64:
        pixels[col] = pixels[col].astype(np.float32)

# Extract the columns that should be binary
for column in [col for col in pixels.columns if "microdissection" in col] + ["Density_Defects", "Density_Lesion"]:
    # Make the column binary
    pixels[column] = pixels[column] > 0.5

    # Remove the Density_ prefix since it is binary now
    pixels.rename(columns={column: column.replace("Density_", "")}, inplace=True)

# Drop the columns that are not needed
pixels.drop(columns=["run", "X3DPositionZ"], inplace=True)

# Define the available stains
stains = ["HES", "MALDI", "BleuAlcian", "PicroSiriusRed", "PANCKm-CD8r"]

# Load the aligned images and downsample them
stained_images = {stain: tiff.imread(f"{path}/{slide}/results/images_aligned/{stain}.ome.tiff")[::downsample, ::downsample]
                  for stain in stains}

# Add the metadata that was in imzML file
metadata = {'chemistry_description': 'MALDI-MSI',
            'software_version': 'Cardinal 3.6'}

experiment_data = {'spectrumType': 'mass spectrum',
                   'instrumentModel': 'Bruker Daltonics flex series',
                   'ionSource': 'matrix-assisted laser desorption ionization',
                   'analyzer': 'time-of-flight',
                   'detectorType': 'microchannel plate detector',
                   'lineScanSequence': 'top down',
                   'scanPattern': 'meandering',
                   'scanType': 'horizontal line scan',
                   'lineScanDirection': 'linescan right left',
                   'pixelSize': '50'}

# Define the scalefactors to help in the visualization after
scalefactors = {**{f'tissue_{stain}_scalef': 1 for stain in stains},
                'spot_diameter_fullres': (150/downsample)*((pixels.x_warped.max() - pixels.x_warped.min()) / len(pixels)),
                'fiducial_diameter_fullres': (200/downsample)*((pixels.x_warped.max() - pixels.x_warped.min()) / len(pixels)),
                'regist_target_img_scalef': 1}


# Define the unstructured data as nested dictionaries
uns = {"spatial": {f'{slide}': {'images': stained_images,
                                'metadata': metadata,
                                'scalefactors': scalefactors}},
       "experiment_data": experiment_data}

coord = {'spatial': pixels[['x_warped', 'y_warped']].values / downsample}

# Create the AnnData object
adata = anndata.AnnData(X=csr_matrix(peaks.values), var=features, obs=pixels, uns=uns, obsm=coord)

# Save the AnnData object to disk with compression
adata.write_h5ad(f"{path}/{slide}/results/adata.h5ad", compression="gzip")