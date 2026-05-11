import splitfolders

# Input directory: Path to your current 'train_augmented' folder
input_folder = "dataset/train_augmented"

# Output directory: Where the new 'train', 'val', and 'test' folders will be created
output_folder = "dataset/final_split"

# Split ratio: (train, validation, test)
# 0.7 = 70%, 0.15 = 15%, 0.15 = 15%
splitfolders.ratio(input_folder, output=output_folder, 
                   seed=1337, ratio=(0.7, 0.15, 0.15))

print("✅ Success! Your dataset is now split into Train, Val, and Test folders.")
