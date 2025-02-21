



import sys
sys.path.append('/home/yasmin/Desktop/CLAM/')  # Percorso del progetto CLAM
import os
import numpy as np
import pandas as pd
import time
from wsi_core.WholeSlideImage import WholeSlideImage
from wsi_core.wsi_utils import StitchPatches
from wsi_core.batch_process_utils import initialize_df

# Percorsi specifici per il dataset
source = '/mnt/DATA/YASMIN/immagini'
save_dir = ''
patch_save_dir = os.path.join(save_dir, 'patches')
os.makedirs(patch_save_dir, exist_ok=True)

# Parametri del patching
patch_size = 512
step_size = 512
max_patches = 50

# Funzione per processare una singola immagine (segmentazione e patching)
def process_single_image(slide_path, save_dir, patch_save_dir, patch_size, step_size, max_patches):
    print(f"Inizio elaborazione di: {slide_path}")
    slide_id = os.path.splitext(os.path.basename(slide_path))[0]

    # Inizializza oggetto WSI
    WSI_object = WholeSlideImage(slide_path)
    width, height = WSI_object.level_dim[0]
    print(f"Dimensioni immagine: {width}x{height}")

    # Patching
    patch_id = 0
    for y in range(0, height, step_size):
        for x in range(0, width, step_size):
            if patch_id >= max_patches:
                print("Raggiunto il limite massimo di patch.")
                return
            patch = WSI_object.read_region((x, y), 0, (patch_size, patch_size)).convert("RGB")
            patch_array = np.array(patch)
            white_ratio = np.mean(patch_array > 240)
            if white_ratio < 0.9:  # Filtra patch vuote o bianche
                patch_filename = f"{slide_id}_patch_{patch_id}.jpg"
                patch.save(os.path.join(patch_save_dir, patch_filename))
                patch_id += 1

    print(f"Processamento completato per {slide_path}. Patch generate: {patch_id}")

# Seleziona solo la prima immagine per test
slides = sorted(os.listdir(source))
slides = [slide for slide in slides if slide.endswith('.tif')]

if slides:
    first_slide = slides[0]
    slide_path = os.path.join(source, first_slide)
    process_single_image(slide_path, save_dir, patch_save_dir, patch_size, step_size, max_patches)
else:
    print("Nessuna immagine trovata nella directory.")
