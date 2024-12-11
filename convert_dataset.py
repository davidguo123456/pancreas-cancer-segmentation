import os
import shutil
import re
from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json

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

    # Move test images
    for file in os.listdir(test_path):
        if re.search(r"_\d{4}\.nii\.gz$", file):
            shutil.move(
                os.path.join(test_path, file),
                os.path.join(dest_images_ts, file)
            )

if __name__ == "__main__":
    base_folder = "UHN-MedImg3D-ML-quiz"  # Adjust this path as needed
    dest_folder = "data/Dataset011_Pancreas"
    organize_files(base_folder, dest_folder)
    generate_dataset_json(dest_folder,
                          channel_names={0: 'CT'},
                          labels={
                            'background': 0,
                            'pancreas': (1,2),
                            'lesion': 2,
                          },
                          regions_class_order=(1,2),
                          num_training_cases=252,
                          file_ending='.nii.gz',
                          overwrite_image_reader_writer='SimpleITKIO',
                          )

