import numpy as np
import cv2
import scipy.ndimage
from skimage.filters import frangi
from skimage import exposure
from medpy.filter.smoothing import anisotropic_diffusion

def resize_volume_and_mask(volume, target_shape=(256, 256, 256)):
    zoom_factors = (
        target_shape[0] / volume.shape[0],
        target_shape[1] / volume.shape[1],
        target_shape[2] / volume.shape[2]
    )
    resized_volume = scipy.ndimage.zoom(volume, zoom_factors, order=1)  # interpolation linéaire
    return resized_volume

def normalize_image(img):
    img = img.astype(np.float32)
    img = (img - np.min(img)) / (np.max(img) - np.min(img) + 1e-8)
    return img


#### Preprocessing volume for Our_Method_with_Filters ####################

def step1_preprocess_img_slice_Our_Method_with_Filters(img_slc):
   
    # 1. Suppression des outliers (valeurs > 3000 HU)
    img_slc = np.where(img_slc > 3000, 0, img_slc)

    # 2. Clipping dans l’intervalle [0, 300] HU
    # img_slc   = np.clip(img_slc, 30, 300) this is the correct one
    img_slc   = np.clip(img_slc, 0, 300)

    # Sauvegarde de l’image originale (pour superposition finale)
    img_original = img_slc.copy()

    # 3. Normalisation simple (valeurs entre 0 et 1)
    img_slc = normalize_image(img_slc)

    # 5. Mise à l’échelle pour CLAHE
    img_slc = img_slc * 255
    img_slc = img_slc.astype('uint8')

    # 6. Application de CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16, 16))
    img_slc = clahe.apply(img_slc)
    img_slc = normalize_image(img_slc)
    
    # 7. Diffusion anisotrope (CED filtering)
    img_ced = anisotropic_diffusion(img_slc, niter=15, kappa=10, gamma=0.1, option=1)
    img_slc = normalize_image(img_ced)

    # 8. Filtrage Frangi (vesselness)
    img_frangi = frangi(img_ced)
    img_frangi = normalize_image(img_frangi)

    # 9. Mise à l’échelle pour affichage [0, 1]
    img_frangi = exposure.rescale_intensity(img_frangi, out_range=(0, 1))

    # 10. Superposition des vaisseaux détectés
    superposed_img = img_original.copy()
    vessel_mask = img_frangi > 0.5
    superposed_img[vessel_mask] = 1.0

    return superposed_img

def preprocess_volume_Our_Method_with_Filters(volume):
    volume_preprocessed = np.zeros_like(volume)
    for i in range(volume.shape[2]):  # boucle sur D (les coupes axiales)
        volume_preprocessed[:, :, i] = step1_preprocess_img_slice_Our_Method_with_Filters(volume[:, :, i])
    volume_preprocessed = normalize_image(volume_preprocessed)
    return volume_preprocessed
 
#### Preprocessing volume for Our_Method_without_Filters ####################

def step1_preprocess_img_slice_Our_Method_without_Filters(img_slc):

    # 1. Suppression des outliers (valeurs > 1200 HU)
    img_slc = np.where(img_slc > 3000, 0, img_slc)

    # 2. Clipping dans l’intervalle [0, 300] HU
    img_slc   = np.clip(img_slc, 0, 300)

    # 3. Normalisation simple (valeurs entre 0 et 1)
    img_slc = normalize_image(img_slc)

    # 5. Mise à l’échelle pour CLAHE
    img_slc = img_slc * 255
    img_slc = img_slc.astype('uint8')

    # 6. Application de CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16, 16))
    img_slc = clahe.apply(img_slc)

    img_slc = normalize_image(img_slc)

    return img_slc

def preprocess_volume_Our_Method_without_Filters(volume):
    volume_preprocessed = np.zeros_like(volume)
    for i in range(volume.shape[2]):  # boucle sur D (les coupes axiales)
        volume_preprocessed[:, :, i] = step1_preprocess_img_slice_Our_Method_without_Filters(volume[:, :, i])
    volume_preprocessed = normalize_image(volume_preprocessed)
    return volume_preprocessed
 
#### Preprocessing volume for 3D-Unet Resized ####################
import numpy as np
import cv2
from skimage.filters import frangi

def step1_preprocess_img_slice_3d_Resized(img_slc):

    # Suppression des outliers
    img_slc = np.where(img_slc > 1200, 0, img_slc)

    # Clipping pour garder les tissus pertinents
    img_slc = np.clip(img_slc, -100, 400)

    # Normalisation Z-score
    mean, std = np.mean(img_slc), np.std(img_slc)
    img_slc = (img_slc - mean) / (std + 1e-8)  # éviter la division par zéro

    # Mise à l'échelle pour CLAHE (0 à 255)
    img_slc = ((img_slc - img_slc.min()) / (img_slc.max() - img_slc.min()) * 255).astype('uint8')

    # Application de CLAHE pour rehausser le contraste
    clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(8,8))
    img_slc = clahe.apply(img_slc)

    img_slc = (img_slc - img_slc.min()) / (img_slc.max() - img_slc.min() + 1e-8)

    return img_slc

def preprocess_volume_3d_Resized(volume):
    volume_preprocessed = np.zeros_like(volume)
    for i in range(volume.shape[2]):  # boucle sur D (les coupes axiales)
        volume_preprocessed[:, :, i] = step1_preprocess_img_slice_3d_Resized(volume[:, :, i])
    volume_preprocessed = normalize_image(volume_preprocessed)
    return volume_preprocessed