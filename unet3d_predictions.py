import os
import numpy as np
import nibabel as nib
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras import models
import torch
import torch.nn as nn
import scipy.ndimage
import subprocess # <- Ajouter cette ligne
import shutil     # <- Ajouter cette ligne
import tempfile   # <- Ajouter cette ligne
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from scipy.ndimage import gaussian_filter, binary_opening, label

import torch
import torch.nn as nn
import torch.nn.functional as F

from monai.metrics import DiceMetric, HausdorffDistanceMetric
from monai.data import Dataset, DataLoader
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd,
    ScaleIntensityRanged, ToTensord, Resized, Orientationd, Spacingd, CropForegroundd
)
import glob

# -------------------- DropBlock 3D --------------------
class DropBlock3D(nn.Module):
    def __init__(self, block_size=3, drop_prob=0.4):
        super().__init__()
        self.block_size = block_size
        self.drop_prob = drop_prob

    def forward(self, x):
        if not self.training or self.drop_prob == 0.0:
            return x
        gamma = self.drop_prob / (self.block_size ** 3)
        mask = (torch.rand_like(x) < gamma).float()
        mask = F.max_pool3d(mask, self.block_size, stride=1, padding=self.block_size // 2)
        return x * (1.0 - mask)

# -------------------- CBAM Block --------------------
class CBAMBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
        self.spatial = nn.Sequential(
            nn.Conv3d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _, _ = x.size()
        y = self.pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1, 1)
        x = x * y
        max_pool = torch.max(x, dim=1, keepdim=True)[0]
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        spatial_att = self.spatial(torch.cat([avg_pool, max_pool], dim=1))
        return x * spatial_att

# -------------------- DoubleConv --------------------
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, num_units=2):
        super().__init__()
        layers = []
        for i in range(num_units):
            conv = nn.Conv3d(
                in_channels if i == 0 else out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False
            )
            nn.init.kaiming_normal_(conv.weight, mode='fan_out', nonlinearity='leaky_relu')
            layers.append(conv)
            layers.append(nn.InstanceNorm3d(out_channels, affine=True))
            layers.append(nn.GroupNorm(8, out_channels))
            layers.append(nn.LeakyReLU(inplace=True))
        self.double_conv = nn.Sequential(*layers)
        self.attn = CBAMBlock(out_channels)

    def forward(self, x):
        x = self.double_conv(x)
        return self.attn(x)

# -------------------- Down --------------------
class Down(nn.Module):
    def __init__(self, in_channels, out_channels, num_units=2):
        super().__init__()
        self.down = nn.Sequential(
            nn.MaxPool3d(2),
            DoubleConv(in_channels, out_channels, num_units),
            DropBlock3D(drop_prob=0.2)
        )

    def forward(self, x):
        return self.down(x)

# -------------------- Up --------------------
class Up(nn.Module):
    def __init__(self, in_channels, out_channels, num_units=2, dropout=0.4):
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False),
            nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False)
        )
        self.dropblock = DropBlock3D(drop_prob=dropout)
        self.conv = DoubleConv(in_channels, out_channels, num_units)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        if x1.shape[2:] != x2.shape[2:]:
            x2 = F.interpolate(x2, size=x1.shape[2:], mode='trilinear', align_corners=False)
        x = torch.cat([x2, x1], dim=1)
        x = self.dropblock(x)
        return self.conv(x)

# -------------------- UNet3D --------------------
class UNet3D(nn.Module):
    def __init__(self, in_channels=1, out_channels=1,
                 channels=(32, 64, 128, 256, 512),
                 num_res_units=2, dropout=0.4):
        super().__init__()

        self.inc = DoubleConv(in_channels, channels[0], num_res_units)
        self.down1 = Down(channels[0], channels[1], num_res_units)
        self.down2 = Down(channels[1], channels[2], num_res_units)
        self.down3 = Down(channels[2], channels[3], num_res_units)
        self.down4 = Down(channels[3], channels[4], num_res_units)

        self.global_context = nn.Conv3d(channels[4], channels[4], kernel_size=3, padding=6, dilation=6)

        self.up1 = Up(channels[4], channels[3], num_res_units, dropout)
        self.up2 = Up(channels[3], channels[2], num_res_units, dropout)
        self.up3 = Up(channels[2], channels[1], num_res_units, dropout)
        self.up4 = Up(channels[1], channels[0], num_res_units, dropout)

        self.proj_x3 = nn.Conv3d(channels[2], channels[3], kernel_size=1)
        self.proj_x2 = nn.Conv3d(channels[1], channels[2], kernel_size=1)
        self.proj_x1 = nn.Conv3d(channels[0], channels[1], kernel_size=1)

        self.outc = nn.Conv3d(channels[0], out_channels, kernel_size=1)
        self.outc2 = nn.Conv3d(channels[1], out_channels, kernel_size=1)
        self.outc3 = nn.Conv3d(channels[2], out_channels, kernel_size=1)

        for m in [self.proj_x3, self.proj_x2, self.proj_x1, self.outc, self.outc2, self.outc3]:
            if hasattr(m, 'weight'):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x5 = self.global_context(x5)

        x3_up = F.interpolate(x3, size=x4.shape[2:], mode='trilinear', align_corners=False)
        x = self.up1(x5, x4 + self.proj_x3(x3_up))

        x2_up = F.interpolate(x2, size=x3.shape[2:], mode='trilinear', align_corners=False)
        x = self.up2(x, x3 + self.proj_x2(x2_up))

        x1_up = F.interpolate(x1, size=x2.shape[2:], mode='trilinear', align_corners=False)
        x_mid2 = self.up3(x, x2 + self.proj_x1(x1_up))

        x_mid1 = self.up4(x_mid2, x1)

        out_main = self.outc(x_mid1)
        aux1 = F.interpolate(self.outc2(x_mid2), size=out_main.shape[2:], mode='trilinear', align_corners=False)
        aux2 = F.interpolate(self.outc3(x), size=out_main.shape[2:], mode='trilinear', align_corners=False)

        out = (out_main + aux1 + aux2) / 3
        return out, aux1, aux2

def resize_volume_and_mask(volume, target_shape=(256, 256, 256)):
    zoom_factors = (
        target_shape[0] / volume.shape[0],
        target_shape[1] / volume.shape[1],
        target_shape[2] / volume.shape[2]
    )
    resized_volume = scipy.ndimage.zoom(volume, zoom_factors, order=1)  # interpolation linéaire
    return resized_volume

def extract_patches_3d(image, patch_size=(256, 256, 64), stride=(256, 256, 64)):
    img_shape = image.shape
    patches_img = []

    for z in range(0, img_shape[2] - patch_size[2] + 1, stride[2]):
        for y in range(0, img_shape[1] - patch_size[1] + 1, stride[1]):
            for x in range(0, img_shape[0] - patch_size[0] + 1, stride[0]):
                patch_img = image[x:x+patch_size[0], y:y+patch_size[1], z:z+patch_size[2]]

                if patch_img.shape == patch_size:
                    patches_img.append((patch_img, (x, y, z)))  # ajouter la position

    return patches_img

def predict_one_patch(model, patch, is_pytorch=False):
    # Normaliser si nécessaire ici
    patch = patch.astype(np.float32) / np.max(patch)  # simple normalization
    
    if is_pytorch:
        # Pour les modèles PyTorch - format: (batch, channels, depth, height, width)
        patch_input = np.expand_dims(patch, axis=0)  # batch dim
        patch_input = np.expand_dims(patch_input, axis=0)  # channel dim
        with torch.no_grad():
            patch_tensor = torch.from_numpy(patch_input).float()
            prediction = model(patch_tensor)
            prediction = prediction.numpy()
    else:
        # Pour les modèles Keras/TensorFlow - format: (batch, depth, height, width, channels)
        patch_input = np.expand_dims(patch, axis=0)  # batch dim
        patch_input = np.expand_dims(patch_input, axis=-1)  # channel dim
        prediction = model.predict(patch_input)
    
    return np.squeeze(prediction)

def reconstruct_volume(patch_predictions, positions, volume_shape):
    result_volume = np.zeros(volume_shape, dtype=np.float32)

    for pred, (x, y, z) in zip(patch_predictions, positions):
        result_volume[x:x+pred.shape[0], y:y+pred.shape[1], z:z+pred.shape[2]] = pred

    return result_volume

def run_3dunet_prediction(test_image_path, image_data, model_choice, file_name):

    if model_choice == "liver":
        # Chemin vers le modèle et l'image test
        model_path = "Models/checkpoint_best.pth"

        output_path = "predicted/predicted_mask.nii.gz"

    # Vérifier si le fichier modèle existe
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Le fichier modèle '{model_path}' n'existe pas. Veuillez vérifier que le modèle est présent.")

    # Charger le modèle
    if model_path.endswith('.pth'):
        # Charger un modèle PyTorch
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        # Vérifier si c'est un dictionnaire (state dict) ou un modèle complet
        if isinstance(checkpoint, dict):
            # Si c'est un state dict, créer un modèle et charger les poids
            print("Chargement d'un modèle nnUNet à partir des poids...")
            model = Simple3DUNet(in_channels=1, out_channels=1)
            
            # Extraire les poids du réseau si c'est un checkpoint nnUNet
            if 'network_weights' in checkpoint:
                network_weights = checkpoint['network_weights']
                print("Checkpoint nnUNet détecté, extraction des poids du réseau...")
            else:
                network_weights = checkpoint
                
            try:
                model.load_state_dict(network_weights)
                print("Poids chargés avec succès!")
            except Exception as e:
                print(f"Erreur lors du chargement des poids: {e}")
                print("Tentative de chargement avec strict=False...")
                model.load_state_dict(network_weights, strict=False)
            model.eval()
        else:
            # Si c'est un modèle complet
            model = checkpoint
            model.eval()
    else:
        # Charger un modèle Keras/TensorFlow
        model = models.load_model(model_path, compile=False)

    # Charger l'image NIfTI
    img_nifti = nib.load(test_image_path)

    target_shape_initial = image_data.shape

    image_data = resize_volume_and_mask(image_data, target_shape=(256, 256, 64))

    # Extraire les patchs
    patches_data = extract_patches_3d(image_data, patch_size=(256,256,64), stride=(256,256,64))
    patches, positions = zip(*patches_data)

    # Prédire chaque patch
    predictions = []
    is_pytorch = model_path.endswith('.pth')
    for patch in tqdm(patches):
        pred = predict_one_patch(model, patch, is_pytorch)
        predictions.append(pred)

    # Reconstruire le volume
    volume_shape = image_data.shape  # (256, 256, 256)
    reconstructed_volume = reconstruct_volume(predictions, positions, volume_shape)

    reconstructed_volume = (reconstructed_volume > 0.5).astype(np.float32)

    reconstructed_volume = resize_volume_and_mask(reconstructed_volume, target_shape=target_shape_initial)

    # Binariser le volume (seuillage à 0.5)
    reconstructed_volume = (reconstructed_volume > 0.5).astype(np.float32)

    # Sauvegarder la prédiction en fichier NIfTI
    pred_nifti = nib.Nifti1Image(reconstructed_volume, affine=img_nifti.affine)
    if not file_name.startswith("pre"):
        nib.save(pred_nifti, output_path)




def run_nnunet_prediction_liver(
    input_file,
    dataset_id = 100,
    output_dir="predicted",
    config="3d_fullres",
    trainer="nnUNetTrainer_500epochs",
    folds="all",
    save_probabilities=False
):
    """
    Run nnU-Net v2 prediction from Python.

    Parameters
    ----------
    inputinput_file_dir : str
        Path to file with test images (nii.gz).
    dataset_id : int
        Dataset ID (e.g., 100 for Dataset100_LIVER).
    output_dir : str
        Path to save predictions.
    config : str
        Network configuration (default: "3d_fullres").
    trainer : str
        Trainer class name (default: "nnUNetTrainer_500epochs").
    folds : str or list
        Which folds to use. "all" = ensemble all trained folds.
    save_probabilities : bool
        If True, also saves softmax probabilities.
    """

    # Make sure output dir exists
    os.makedirs(output_dir, exist_ok=True)
    device=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("DEVICE: ", device)
    # Initialize predictor
    predictor = nnUNetPredictor(

        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=True,
        device=device,
        verbose=True,
        verbose_preprocessing=True,
        allow_tqdm=True
    )

    # Initialize nnU-Net with your model
    predictor.initialize_from_trained_model_folder(
        model_training_output_dir=os.path.join(
            os.environ["nnUNet_results"],
            f"Dataset{dataset_id}_LIVER",
            f"{trainer}__nnUNetPlans__{config}"
        ),
        use_folds=folds,
        checkpoint_name="checkpoint_best.pth"
    )
    file_name = input_file.split("/")[-1].split(".")[0]
    # Run prediction
    predictor.predict_from_files(
        [
            [
                input_file
            ]
        ],
        "predicted",
        save_probabilities=save_probabilities,
        overwrite=True,
        num_processes_preprocessing=1,
        num_processes_segmentation_export=1
    )

    # base_name = os.path.basename(input_file)
    predicted_path = os.path.join(output_dir, "li.nii.gz")
    if os.path.exists(predicted_path):
        if os.path.exists( os.path.join(output_dir, "predicted_mask.nii.gz") ):
            os.remove(os.path.join(output_dir, "predicted_mask.nii.gz") ) 
        os.rename(predicted_path, os.path.join(output_dir, "predicted_mask.nii.gz"))
    else:
        raise FileNotFoundError(f"Expected nnU-Net output not found: {predicted_path}")

    keep_extensions = [".nii.gz"]

    for f in os.listdir("predicted"):
        if not any(f.endswith(ext) for ext in keep_extensions):
            os.remove(os.path.join("predicted", f))



    print(f"[OK] Predictions saved to {output_dir}/predicted_mask.nii.gz")


def save_prediction_as_nii(pred_mask, affine, output_full_path):
    """Sauvegarde le masque prédit au format NIfTI"""
    pred_nii = nib.Nifti1Image(pred_mask.astype(np.float32), affine)
    nib.save(pred_nii, output_full_path) # Use the provided full path

def create_model(device='cuda'):
    model = UNet3D().to(device)
    test_input = torch.rand(2, 1, 128, 128, 64).to(device)
    outputs = model(test_input)
    print("Vérification des dimensions:")
    print(f"Input shape: {test_input.shape}")
    for i, out in enumerate(outputs):
        print(f"Output {i} shape: {out.shape}")
        assert out.shape[2:] == test_input.shape[2:], f"Dimension mismatch in output {i}: {out.shape[2:]} vs {test_input.shape[2:]}"
    return model

def load_model(model_path, device):
    model_files = glob.glob(model_path)
    if not model_files:
        raise FileNotFoundError(f"Aucun modèle trouvé avec le pattern: {model_path}")
    
    model = create_model(device)
    model.load_state_dict(torch.load(sorted(model_files)[-1], map_location=device))
    model.eval()
    return model

def get_test_transforms_heart():
    return Compose([
        LoadImaged(keys=["image"], reader="NibabelReader"), # Removed "label"
        EnsureChannelFirstd(keys=["image"]), # Removed "label"
        Resized(keys=["image"], spatial_size=(128, 128, 64), mode=["trilinear"]), # Removed "label", "nearest"
        ScaleIntensityRanged(keys=["image"], a_min=0, a_max=2100, b_min=0.0, b_max=1.0, clip=True),
        ToTensord(keys=["image"]) # Removed "label"
    ])

def get_test_transforms_spleen():
    return Compose([
        LoadImaged(keys=["image"]), 
        EnsureChannelFirstd(keys=["image"]),
        ScaleIntensityRanged(keys=["image"], a_min=-57, a_max=164, b_min=0.0, b_max=1.0, clip=True),
        CropForegroundd(keys=["image"], source_key="image", allow_smaller=True),
        Orientationd(keys=["image"], axcodes="RAS"),
        Spacingd(keys=["image"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear")),
        Resized(keys=["image"], spatial_size=(160,160,160) , mode=("trilinear")),
        ToTensord(keys=["image"]) # Added ToTensord for spleen transforms as well.
    ])

def postprocess_mask(pred_mask, which):
    """
    Applique le prétraitement du volume segmenté pour la rate:
    1. Extraction de la plus grande composante connexe
    2. Filtrage morphologique (ouverture)
    3. Lissage gaussien
    """
    if which == "heart":
        structure_element_size = 3
    else:
        structure_element_size = 2
    try:
        # 1. Extraction de la plus grande composante connexe
        labeled_mask, num_labels = label(pred_mask)
        if num_labels > 0:
            # Trouver la plus grande composante
            component_sizes = np.bincount(labeled_mask.ravel())
            if len(component_sizes) > 1:
                largest_component = np.argmax(component_sizes[1:]) + 1
                pred_mask = (labeled_mask == largest_component).astype(np.uint8)
        
        # 2. Filtrage morphologique (ouverture) - élément plus petit pour la rate
        structure = np.ones(
            (structure_element_size, 
             structure_element_size, 
             structure_element_size), 
            dtype=bool
        )
        pred_mask = binary_opening(pred_mask, structure=structure).astype(np.uint8)
        
        # 3. Lissage gaussien plus prononcé pour la forme lisse de la rate
        pred_smoothed = gaussian_filter(
            pred_mask.astype(np.float32), 
            sigma=0.8
        )
        
        return pred_smoothed
    
    except Exception as e:
        print(f"Erreur de post-traitement: {str(e)}")
        return pred_mask

def predict_from_model(model_path, device, input_file, which):
    model = load_model(model_path, device)
    
    test_files = [{"image": input_file}]
    if which == "heart":
        transform = get_test_transforms_heart()
    else:
        transform = get_test_transforms_spleen()
    
    test_ds = Dataset(data=test_files, transform=transform)
    test_loader = DataLoader(test_ds, batch_size=1, num_workers=0)
    
    final_output_path = "" # Renamed from output_path to avoid confusion with parameter in save_prediction_as_nii
    for batch in tqdm(test_loader, desc="Évaluation"):
        case_id = os.path.basename(input_file).split('.')[0] 
        inputs = batch["image"].to(device)
        
        outputs = model(inputs)
        main_output = outputs[0] if isinstance(outputs, (tuple, list)) else outputs
        preds = (torch.sigmoid(main_output) > 0.5).float()
        
        pred_mask = preds.squeeze().cpu().numpy()
        
        affine = batch["image_meta_dict"]["affine"][0].numpy() if "image_meta_dict" in batch else nib.load(input_file).affine

        original_nifti = nib.load(input_file)
        original_shape = original_nifti.get_fdata().shape
        
        zoom_factors = (
            original_shape[0] / pred_mask.shape[0],
            original_shape[1] / pred_mask.shape[1],
            original_shape[2] / pred_mask.shape[2]
        )
        resized_pred_mask = scipy.ndimage.zoom(pred_mask, zoom_factors, order=0)

        resized_pred_mask = postprocess_mask(resized_pred_mask, which)

        output_dir_for_save = "predicted" # Consistent output directory
        os.makedirs(output_dir_for_save, exist_ok=True)
        final_output_path = os.path.join(output_dir_for_save, f"{case_id}_predicted_mask.nii.gz")
        save_prediction_as_nii(resized_pred_mask, affine, final_output_path)
    
    return final_output_path

def run_nnunet_prediction_heart_spleen(input_file, which="heart", output_dir="predicted", output_filename="predicted_mask.nii.gz"):
    device=torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    if which == "heart":
        model_weights_path = "Models/heart.pth"
    else:
        model_weights_path = "Models/spleen_model.pth"

    final_output_path = predict_from_model(model_weights_path, device, input_file, which)
    print(f"[OK] Predicted mask saved to {final_output_path}")

    
    return final_output_path