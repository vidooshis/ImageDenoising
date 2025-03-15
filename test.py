import os
import cv2
import numpy as np

import tensorflow as tf
from tensorflow.keras.models import load_model

from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

def load_denoise_model(model_path="./denoise_model.h5"):
    print("Loading model...")
    model = load_model(model_path)
    print("Model loaded successfully!")
    model.summary()
    return model

def denoise_image(model, noisy_img):
    input_batch = np.expand_dims(noisy_img, axis=0)
    denoised_batch = model.predict(input_batch)
    denoised = denoised_batch[0]
    denoised = np.clip(denoised, 0, 1)
    return denoised

def calculate_metrics(gt, denoised):
    psnr_value = psnr(gt, denoised, data_range=1.0)
    ssim_value = ssim(gt, denoised, data_range=1.0, channel_axis=2)
    return psnr_value, ssim_value

def process_folder(folder_path, model):
    gt_file, noisy_file = None, None
    for f in os.listdir(folder_path):
        f_lower = f.lower()
        if 'gt' in f_lower and f_lower.split('.')[-1] in ['png', 'tiff', 'tif']:
            gt_file = os.path.join(folder_path, f)
        elif 'noisy' in f_lower and f_lower.split('.')[-1] in ['png', 'tiff', 'tif']:
            noisy_file = os.path.join(folder_path, f)
    if not gt_file or not noisy_file:
        return None

    try:
        gt = cv2.imread(gt_file, cv2.IMREAD_UNCHANGED).astype(np.float32)
        noisy = cv2.imread(noisy_file, cv2.IMREAD_UNCHANGED).astype(np.float32)

        if gt.shape != noisy.shape:
            print(f"Dimension mismatch in {folder_path}: GT {gt.shape}, Noisy {noisy.shape}")
            return None
        
        gt = cv2.resize(gt, (256, 256), interpolation=cv2.INTER_CUBIC)
        noisy = cv2.resize(noisy, (256, 256), interpolation=cv2.INTER_CUBIC)

        if len(gt.shape) == 2:
            gt = cv2.cvtColor(gt, cv2.COLOR_GRAY2RGB)
            noisy = cv2.cvtColor(noisy, cv2.COLOR_GRAY2RGB)

        max_val = 65535.0 if np.max(gt) > 255 else 255.0
        gt = (gt / max_val).clip(0, 1)
        noisy = (noisy / max_val).clip(0, 1)

        if gt.shape[-1] == 4:
            gt = gt[..., :3]
        if noisy.shape[-1] == 4:
            noisy = noisy[..., :3]

        denoised = denoise_image(model, noisy)

        psnr_val, ssim_val = calculate_metrics(gt, denoised)
        return psnr_val, ssim_val
    except Exception as e:
        print(f"Error in {folder_path}: {str(e)}")
        return None

def evaluate_dataset(dataset_path, model_path="./denoise_model.h5"):
    model = load_denoise_model(model_path)
    total_folders = 0
    valid_pairs = 0
    psnr_values = []
    ssim_values = []
    
    print(f" Scanning dataset at: {dataset_path}\n")

    for root, dirs, files in os.walk(dataset_path):
        for dir_name in dirs:
            total_folders += 1
            folder_path = os.path.join(root, dir_name)
            metrics = process_folder(folder_path, model)
            
            if metrics:
                valid_pairs += 1
                psnr_val, ssim_val = metrics
                psnr_values.append(psnr_val)
                ssim_values.append(ssim_val)
                
                print(f" {dir_name}")
                print(f" PSNR: {psnr_val:.2f} dB | SSIM: {ssim_val:.4f}")

    print("\nFinal Results:")
    print(f"Total folders scanned: {total_folders}")
    print(f"Valid pairs processed: {valid_pairs}")
    
    if valid_pairs > 0:
        print(f"Average PSNR: {np.mean(psnr_values):.2f} dB")
        print(f"Average SSIM: {np.mean(ssim_values):.3f}")
    else:
        print("No valid pairs found")

if __name__ == "__main__":
    DATASET_PATH = "./SIDD_Medium_Raw/Data"
    MODEL_PATH = "./denoise_model.h5"  
    evaluate_dataset(DATASET_PATH, MODEL_PATH)