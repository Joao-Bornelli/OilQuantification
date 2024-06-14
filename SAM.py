import cv2
import numpy as np
from ultralytics import SAM
import os

# Initialize the SAM predictor
predictor = SAM("sam_b.pt")

# Set paths for images and labels
imagepath = r'C:\Users\joaobo\Documents\OilVisualization\Images\AllResized'
labelpath = r'C:\Users\joaobo\Documents\OilVisualization\runs\Labels'

# Get list of image and label names
imagenames = os.listdir(imagepath)
labelnames = os.listdir(labelpath)

# List to store points clicked on the image
points = ()

def normalize_contours(contours, image_shape):
    """
    Normalize the coordinates of contours based on image dimensions.
    """
    h, w = image_shape[:2]
    norm_contours = []
    for contour in contours:
        if len(contour) < 8:
            continue
        norm_contour = [0]
        for point in contour:
            x, y = point[0]
            normalized_x = x / w
            normalized_y = y / h
            norm_contour.extend([normalized_x, normalized_y])
        norm_contours.append(norm_contour)
    return norm_contours

def mouse_callback(event, x, y, flags, param):
    """
    Mouse callback function to record points clicked.
    """
    global points
    if event == cv2.EVENT_LBUTTONDOWN:  # Left mouse button click
        points = (x, y)
        print(points)

def process_image(image):
    """
    Process an image: read, convert to grayscale, equalize histogram,
    and set up the display window with mouse callback.
    """
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    equalized_image = cv2.equalizeHist(gray_image)
    three_channel_image = cv2.cvtColor(equalized_image, cv2.COLOR_GRAY2BGR)
    
    cv2.namedWindow('image')
    cv2.setMouseCallback('image', mouse_callback)
    
    return three_channel_image

def draw_contours_on_image(image, results):
    """
    Draw contours on the image based on the results.
    """
    all_contours = []
    for result in results:
        if result[0].masks is not None:
            for mask in result[0].masks.data:
                mask = mask.cpu().numpy()
                mask = (mask * 255).astype(np.uint8)
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(image, contours, -1, (0, 255, 0), 1)
                all_contours.extend(contours)
    return image, all_contours

def save_normalized_contours(image_shape, all_contours, label_path, imagename):
    """
    Normalize and save contours to a text file.
    """
    normalized_contours = normalize_contours(all_contours, image_shape)
    with open(os.path.join(label_path, imagename + '.txt'), 'a') as file:
        for contour in normalized_contours:
            contour_str = ' '.join(map(str, contour))
            file.write(contour_str + '\n')

def main():
    """
    Main function to iterate through images, process them, and handle user input.
    """
    global points
    for imagename in imagenames:
        if imagename + '.txt' in labelnames:
            continue
        print(imagename)
        image = cv2.imread(os.path.join(imagepath, imagename))
        
        if image is None:
            print(f"Failed to load image {imagename}")
            continue
        
        three_channel_image = process_image(image)

        all_contours = []
        results = []
        display_image = three_channel_image.copy()
        new = False

        while True:
            if new:
                result = predictor(three_channel_image, points=[points])
                results.append(result)
                display_image = three_channel_image.copy()
                display_image, new_contours = draw_contours_on_image(display_image, results)
                
                # Avoid duplicating contours
                all_contours.extend(new_contours)
                
                new = False

            cv2.imshow('image', display_image)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('n'):
                new = True
            elif key == ord('q'):
                break
            elif key == ord('d'):
                if results:
                    results.pop()

        cv2.destroyAllWindows()

        if all_contours:
            save_normalized_contours(image.shape, all_contours, labelpath, imagename)

if __name__ == "__main__":
    main()



# import cv2
# import numpy as np
# from torch import normal
# from ultralytics import SAM
# import os

# predictor = SAM("sam_b.pt")

# imagepath = r'C:\Users\joaobo\Documents\OilVisualization\Images\AllResized'
# imagenames = os.listdir(imagepath)#\Frame_V3_17640.jpg'

# labelpath = r'C:\Users\joaobo\Documents\OilVisualization\runs\Labels'
# labelnames = os.listdir(labelpath)

# def normalizeContours(contours, imageShape):
#         h, w = imageShape[:2]
#         normContours = []
#         for contour in contours:
#             for lines in contour:
#                 if len(lines) < 8:
#                     continue
#                 normContour = []
#                 normContour.append(0)
#                 for point in lines:
#                     x, y = point[0]
#                     normalized_x = x / w
#                     normalized_y = y / h
#                     normContour.append(normalized_x)
#                     normContour.append(normalized_y)
#                 normContours.append(normContour)
#         return normContours
# points = []

# def mouse_callback(event, x, y, flags, param):
#     if event == cv2.EVENT_LBUTTONDOWN:  # Evento de clique com o botão esquerdo do mouse
#         points.append((x, y))
#         print(points)

# for imagename in imagenames:
#     if imagename+'.txt' in labelnames:
#         continue
#     print(imagename)
#     image = cv2.imread(os.path.join(imagepath,imagename))
#     gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     equalized_image = cv2.equalizeHist(gray_image)

#     three_channel_image = cv2.cvtColor(equalized_image, cv2.COLOR_GRAY2BGR)


#     allcontours = []


#     results = []



    
            
    
#     cv2.namedWindow('image')
#     cv2.setMouseCallback('image', mouse_callback)
#     display_image = three_channel_image.copy()
#     new = False
#     while True:
#         if new:


#             result = predictor(three_channel_image, points=points)
#             results.append(result)

#             cv2.destroyAllWindows()
#             display_image = three_channel_image.copy()


#             if results:
#                 for result in results:
#                     if result[0].masks is not None:
#                         for mask in result[0].masks.data:
#                             mask = mask.cpu().numpy()
#                             mask = (mask * 255).astype(np.uint8)
#                             contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#                             cv2.drawContours(display_image, contours, -1, (0, 255, 0), 1)
#                 allcontours.append(contours)
#             new = False

#         cv2.imshow('image', display_image)
        
        

#         key = cv2.waitKey(1) & 0xFF
#         if key == ord('n'):
#             new = True
#         elif key == ord('q'):
#             break
#         elif key == ord('d'):
#             results = results[:-1]
#             points = points[:-2]
            

#     cv2.destroyAllWindows()


#     if allcontours:
#         normalized_contours = normalizeContours(allcontours, image.shape)
#         with open(os.path.join(r'C:\Users\joaobo\Documents\OilVisualization\runs\Labels', imagename + '.txt'), 'a') as file:
#             for contour in normalized_contours:
#                 string = str(contour)
#                 string = string.replace('[','').replace(']','').replace(', ',' ')
#                 file.write(string + '\n')