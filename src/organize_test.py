import os
import shutil

# Define paths
test_path = "../dataset/ASL_Alphabet_Dataset/asl_alphabet_test"
organized_test_path = "../dataset/ASL_Alphabet_Dataset/test_organized"
alphabets = [chr(i) for i in range(65, 91)]  # A-Z

# Create the organized test folder
os.makedirs(organized_test_path, exist_ok=True)

# Create subfolders for each alphabet
for alphabet in alphabets:
    os.makedirs(os.path.join(organized_test_path, alphabet), exist_ok=True)

# Function to extract label from filename
def get_label_from_filename(filename):
    # Assuming the first character or a prefix indicates the label (e.g., 'A_test.jpg')
    for alphabet in alphabets:
        if filename.upper().startswith(alphabet):
            return alphabet
    return None  # Return None if no label is found

# Organize images
print("Organizing test images...")
for img_name in os.listdir(test_path):
    img_path = os.path.join(test_path, img_name)
    if not os.path.isfile(img_path):  # Skip if not a file
        continue
    
    label = get_label_from_filename(img_name)
    if label is None:
        print(f"Warning: Could not determine label for {img_name}, skipping...")
        continue
    
    # Move the image to the corresponding subfolder
    dest_folder = os.path.join(organized_test_path, label)
    dest_path = os.path.join(dest_folder, img_name)
    shutil.move(img_path, dest_path)
    print(f"Moved {img_name} to {dest_folder}")

print("Organization complete!")