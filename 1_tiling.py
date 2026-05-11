import os
import shutil
from PIL import Image

def process_all_folders(root_dir, tile_size=224, stride=150):
    root_dir = os.path.abspath(root_dir)
    # We only want to process these three motifs
    target_motifs = ['tampuk_manggis', 'pucuk_rebung', 'bunga_pecah_lapan']
    
    for split in ['train', 'val']:
        split_path = os.path.join(root_dir, split)
        output_root = os.path.join(root_dir, f"{split}_tiles")

        if not os.path.exists(split_path):
            print(f"⚠️  Skipping {split}: Folder not found at {split_path}")
            continue

        # Clean old tiles directory completely for a fresh start
        if os.path.exists(output_root):
            shutil.rmtree(output_root)
        os.makedirs(output_root, exist_ok=True)

        for motif_folder in target_motifs:
            input_folder = os.path.join(split_path, motif_folder)
            
            # Check if the raw data folder exists for this motif
            if not os.path.exists(input_folder):
                print(f"❌ Missing raw folder: {input_folder}. Please create it!")
                continue

            output_folder = os.path.join(output_root, motif_folder)
            os.makedirs(output_folder, exist_ok=True)

            print(f"✂️  Processing motif: {motif_folder} in {split}...")

            file_count = 0
            tile_count_total = 0
            
            for filename in os.listdir(input_folder):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.webp')):
                    img_path = os.path.join(input_folder, filename)
                    
                    try:
                        img = Image.open(img_path).convert("RGB")
                        w, h = img.size

                        if w < tile_size or h < tile_size:
                            scale = tile_size / min(w, h)
                            img = img.resize((int(w * scale) + 1, int(h * scale) + 1), Image.Resampling.LANCZOS)
                            w, h = img.size

                        base_name = os.path.splitext(filename)[0]
                        
                        current_file_tiles = 0
                        for y in range(0, h - tile_size + 1, stride):
                            for x in range(0, w - tile_size + 1, stride):
                                box = (x, y, x + tile_size, y + tile_size)
                                tile = img.crop(box)
                                tile.save(os.path.join(output_folder, f"{base_name}_t{current_file_tiles}.jpg"))
                                current_file_tiles += 1
                                tile_count_total += 1
                        file_count += 1
                    except Exception as e:
                        print(f"  ❌ Error processing {filename}: {e}")

            print(f"  ✅ Done: {file_count} images turned into {tile_count_total} tiles.")

if __name__ == "__main__":
    process_all_folders("dataset")
    print("\n🎉 All target motifs tiled successfully. You can now run 2_train.py!")