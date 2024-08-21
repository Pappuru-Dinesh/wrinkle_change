from ultralytics import YOLO
import os
from tqdm import tqdm
from PIL import Image
import cv2

def is_valid_image_pillow(file_name):
    try:
        with Image.open(file_name) as img:
            img.verify()
            return True
    except (IOError, SyntaxError):
        return False

def get_cropped_image(model, image_path, confidence=0.5):
        # Model result
        # parts_dict = {0: 'lips_n_up', 1: 'cheek', 2: 'chin', 3: 'submentum', 4: 'neck'}
        part_id = 1
        model_result = model.predict(source=image_path, conf=confidence, classes=[part_id], verbose=False)

        # cropped image
        for result in model_result:
            shapes = result.boxes.xyxy
            if len(shapes) != 0:
                for shape in shapes:
                    x,y,w,h = [int(i.item()) for i in shape]
                    img = cv2.imread(image_path)
                    img = img[y:h, x:w]
                    continue
            else:
                print("unable to extract image, returning None")
                img = None
                x,y,w,h = [None for i in range(4)]


        return(img, (x,y,w,h))

class extract_images:
    def __init__(self, folder_path):
        # get root directory

        # verify if folder exists
        self.folder_path = folder_path.replace(os.sep, '/')
        if not os.path.isdir(self.folder_path):
            raise OSError("Image folder does not exists, Please update the path and try again", folder_path)
        self.folder_name = self.folder_path.split('/')[-1]
        print(f'Image folder: {self.folder_name}')
        
        self.extract_folder = "Extracted_"+self.folder_name
        self.ext_folder_path = os.path.join(('/').join(self.folder_path.split('/')[:-1]), self.extract_folder)
        self.ext_folder_path = self.ext_folder_path.replace(os.sep, '/')
        # Create extraction folder
        try:  
            os.mkdir(self.ext_folder_path)  
        except OSError as error:  
            print(error)
            print("Folder already exists, continue")
        
        print(f'Extracted image folder: {self.extract_folder}')

    def extract_cheek_images(self, model=''):
        # Get all the files
        self.image_files = os.listdir(self.folder_path)
        # Filtering only the files.
        self.image_files = [f for f in self.image_files if os.path.isfile(self.folder_path+'/'+f)]
        # Verify valid image files
        print(f"Verifying valid image files")
        self.image_files = [f for f in tqdm(self.image_files) if is_valid_image_pillow(self.folder_path+'/'+f)]
        print(f"Valid image files: {len(self.image_files)}")

        # Load yolo model
        roi_face = self.get_yolo()

        # Extract cheek images
        print("Extracting images")
        for filename in tqdm(self.image_files):
            file_with_path_src = os.path.join(self.folder_path, filename)
            cheek_img, coord = get_cropped_image(roi_face, file_with_path_src)
            
            if cheek_img is None:
                cheek_img, coord = get_cropped_image(roi_face, file_with_path_src, 0.3)
            
            if cheek_img is not None:
                #image save
                file_with_path_dst = os.path.join(self.ext_folder_path, filename)
                x,y,w,h = coord
                org_image = cv2.imread(file_with_path_src)
                cheek_img = org_image[y:h, x:w]
                cheek_img = cv2.resize(cheek_img, (256,256))
                cv2.imwrite(file_with_path_dst, cheek_img)
            else:
                # unable to extract image
                print(f"Unable to extract image: {filename}")
        
        return True
    
    def get_yolo(self):
        self.model_location = r"C:\Users\dpappuru\OneDrive - Cytrellis Biosystems\Desktop\wrinkle_change\wrinkle_change\models\roi_face_yolo.pt"
        self.model_location = self.model_location.replace(os.sep, '/')
        return YOLO(self.model_location)