import os
import shutil
import re

def organize_files(base_folder, dest_folder):
    # Define paths for the source folders
    train_path = os.path.join(base_folder, "train")
    test_path = os.path.join(base_folder, "test")
    validation_path = os.path.join(base_folder, "validation")

    # Define paths for the destination folders
    dest_images_tr = os.path.join(dest_folder, "imagesTr")
    dest_images_ts = os.path.join(dest_folder, "imagesTs")
    dest_images_tv = os.path.join(dest_folder, "imagesVa")
    dest_labels_tr = os.path.join(dest_folder, "labelsTr")
    dest_labels_tv = os.path.join(dest_folder, "labelsVa")

    for path in [dest_images_tr, dest_images_ts, dest_images_tv, dest_labels_tr, dest_labels_tv]:
        os.makedirs(path, exist_ok=True)

    # Move training images
    for subtype in os.listdir(train_path):
        subtype_path = os.path.join(train_path, subtype)
        if os.path.isdir(subtype_path):
            for file in os.listdir(subtype_path):
                if re.search(r"_\d{4}\.nii\.gz$", file):
                    shutil.move(
                        os.path.join(subtype_path, file),
                        os.path.join(dest_images_tr, file)
                    )
                elif file.endswith(".nii.gz"):
                    shutil.move(
                        os.path.join(subtype_path, file),
                        os.path.join(dest_labels_tr, file)
                    )

    # Move validation images and labels
    for subtype in os.listdir(validation_path):
        subtype_path = os.path.join(validation_path, subtype)
        if os.path.isdir(subtype_path):
            for file in os.listdir(subtype_path):
                if re.search(r"_\d{4}\.nii\.gz$", file):
                    shutil.move(
                        os.path.join(subtype_path, file),
                        os.path.join(dest_images_tv, file)
                    )
                elif file.endswith(".nii.gz") and not file.endswith("_\d{4}.nii.gz"):
                    shutil.move(
                        os.path.join(subtype_path, file),
                        os.path.join(dest_labels_tv, file)
                    )

    # Move test images
    for file in os.listdir(test_path):
        if re.search(r"_\d{4}\.nii\.gz$", file):
            shutil.move(
                os.path.join(test_path, file),
                os.path.join(dest_images_ts, file)
            )

if __name__ == "__main__":
    base_folder = "UHN-MedImg3D-ML-quiz"  # Adjust this path as needed
    dest_folder = "nnUNet_raw/Dataset011_Pancreas"
    organize_files(base_folder, dest_folder)

