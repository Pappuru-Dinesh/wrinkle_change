import extract_cheeks as id_c
import os

folder_path = r"C:\Users\dpappuru\OneDrive - Cytrellis Biosystems\Desktop\wrinkle_change\dataset\image_files"

extraction = id_c.extract_images(folder_path).extract_cheek_images()

if extraction:
    print("All valid cheek images extracted")