import os
from PIL import Image
from torchvision import transforms
from tqdm import tqdm  # Install this for a nice progress bar

# 1. SETTINGS
INPUT_ROOT = "dataset/train"  # Your current training data
OUTPUT_ROOT = "dataset/train_augmented"
MULTIPLIER = 10  # Turn 1 image into 10

# 2. DEFINE THE AUGMENTATION "RECIPE"
augment_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.RandomRotation(degrees=45),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    # Optional: adds a bit of perspective shift for "real-world" cloth folds
    transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
])

def enlarge_dataset():
    if not os.path.exists(INPUT_ROOT):
        print(f"❌ Error: {INPUT_ROOT} not found!")
        return

    # Loop through each motif folder (pucuk_rebung, etc.)
    for motif in os.listdir(INPUT_ROOT):
        motif_path = os.path.join(INPUT_ROOT, motif)
        if not os.path.isdir(motif_path): continue

        save_path = os.path.join(OUTPUT_ROOT, motif)
        os.makedirs(save_path, exist_ok=True)

        print(f"⚙️  Enlarging motif: {motif}...")
        
        images = [f for f in os.listdir(motif_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        for img_name in tqdm(images):
            img_path = os.path.join(motif_path, img_name)
            original_img = Image.open(img_path).convert("RGB")
            
            # Save the original first
            original_img.save(os.path.join(save_path, f"orig_{img_name}"))
            
            # Generate the 10 augmented versions
            for i in range(MULTIPLIER - 1):
                augmented_img = augment_transform(original_img)
                name_base = os.path.splitext(img_name)[0]
                augmented_img.save(os.path.join(save_path, f"aug_{i}_{name_base}.jpg"))

if __name__ == "__main__":
    enlarge_dataset()
    print(f"\n✅ Success! Your dataset is now 10x larger in {OUTPUT_ROOT}")
