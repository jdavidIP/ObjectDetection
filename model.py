from ultralytics import YOLO
import cv2 as cv
import numpy as np
import os
import shutil
import time
from ultralytics.utils.plotting import Annotator


def resize_if_too_small(image, min_height=500, min_width=500):
    height, width, _ = image.shape
    if height < min_height or width < min_width:
        aspect_ratio = width / height
        new_width = min_width if width > height else int(min_height * aspect_ratio)
        new_height = min_height if height > width else int(min_width / aspect_ratio)
        resized_image = cv.resize(image, (new_width, new_height))
        return resized_image
    else:
        return image

def divide_and_detect(input_image_path, output_directory="Resources/Sections"):
    img_org = cv.imread(input_image_path)
    img = resize_if_too_small(img_org)

    img_copy = img.copy()

    height, width, channels = img.shape
    # Check if width and height are higher than 1500
    if width > 1200 and height > 1200:
        H_SIZE = height // 416
        W_SIZE = width // 416
    else:
        # If not, set a single section
        H_SIZE = 1
        W_SIZE = 1

    model1 = YOLO(r'minas.pt')
    model2 = YOLO(r'best.pt')

    assembled_images = []  # 2D list to store detected images in order

    # Define a color mapping dictionary for different classes
    class_color_mapping = {
        "class_0": (255, 0, 0),  # Red
        "class_1": (128, 0, 128),  # Green
        "class_2": (0, 0, 255),  # Blue
        # Add more classes and colors as needed
    }

    for ih in range(H_SIZE):
        assembled_row = []  # List to store images in the current row
        for iw in range(W_SIZE):
            x = width / W_SIZE * iw
            y = height / H_SIZE * ih
            h = (height / H_SIZE)
            w = (width / W_SIZE)

            cropped_img = img[int(y):int(y + h), int(x):int(x + w)]

            cv.imwrite("Resources/cropped.jpg", cropped_img)

            img2 = cv.imread(r'Resources/cropped.jpg')

            annotator = Annotator(img2)

            # Capture detection results
            result1 = model1.predict(source=cropped_img, show=False, conf=0.6, save=True)
            result2 = model2.predict(source=cropped_img, show=False, conf=0.6, save=True)

            coordinates1 = []
            class_names1 = []
            
            for r in result1:
                boxes = r.boxes
                i = 0
                for box in boxes:
                    coordinates1.append(box.xyxy[0])  # Assuming Ultralytics result object has xyxy attribute
                    class_names1.append(box.cls)
                    class_color = class_color_mapping.get(f"class_{int(class_names1[i])}", (255, 0, 0))
                    annotator.box_label(coordinates1[i], model1.names[int(class_names1[i])], class_color)
                    i += 1

            coordinates2 = []
            class_names2 = []
            
            for r2 in result2:
                boxes2 = r2.boxes
                i = 0
                for box2 in boxes2:
                    coordinates2.append(box2.xyxy[0])
                    class_names2.append(box2.cls)
                    class_color = class_color_mapping.get(f"class_{int(class_names2[i])}", (255, 0, 0))
                    annotator.box_label(coordinates2[i], model2.names[int(class_names2[i])], class_color)
                    i += 1

            source_path = os.path.join(r'runs\detect\predict', 'image0.jpg')

            cv.imwrite(source_path, annotator.result())

            NAME = str(time.time())

            output_path = os.path.join(output_directory, f"{ih}_{iw}_model1_{NAME}.png")

            shutil.copyfile(source_path, output_path)

            assembled_row.append(cv.imread(output_path))

        assembled_images.append(assembled_row)
        
    img = img_copy

    return assembled_images

def assemble_images(assembled_images):
    # Concatenate rows to reconstruct the original image
    reconstructed_image = np.concatenate([np.concatenate(row, axis=1) for row in assembled_images], axis=0)
    return reconstructed_image

def image_processing(img_path):
    op_img = cv.imread(img_path)

    assembled_images = divide_and_detect(img_path)

    result_image = assemble_images(assembled_images)

    cv.imwrite(r'Resources\Results\result.jpg', result_image)

    shutil.rmtree(r'runs\detect')
    shutil.rmtree(r'Resources/Sections') 
    os.makedirs(r'Resources/Sections')

    return result_image