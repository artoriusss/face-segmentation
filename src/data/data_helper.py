import numpy as np
import cv2
import dlib
import os

class DataHelper():
    def is_frontal_face(self, shape):
        """
            Check if the face is "anfas" (frontal face) by comparing the distances between points
            on the left and right sides of the face to the center point (nose tip, for example).
            Args:
                shape (dlib.full_object_detection): dlib Shape object
            Returns:
                bool: True if the face is frontal, False otherwise
        """
        nose_tip = shape.part(30)
        left_distances = [np.linalg.norm(np.array([shape.part(i).x - nose_tip.x, shape.part(i).y - nose_tip.y])) for i in range(1, 8)]
        right_distances = [np.linalg.norm(np.array([shape.part(16-i).x - nose_tip.x, shape.part(16-i).y - nose_tip.y])) for i in range(1, 8)]
        diff = np.abs(np.array(left_distances) - np.array(right_distances))

        return np.mean(diff) < 10  

    def get_bouding_boxes(self, rects, img_shape): 
        """
            Get the bounding boxes of the faces in the image
            Args:
                rects (dlib.rectangles): dlib rectangles object
            Returns:
                tuple: Tuple of bounding boxes
        """
        x1, y1, x2, y2 = (int(value) for value in rects)

        x_expansion = min(20, x1)
        y_expansion = min(20, y1)
        x2_expansion = min(20, img_shape[0] - x2)
        y2_expansion = min(20, img_shape[1] - y2)

        # Expand coordinates
        x1 -= x_expansion
        y1 -= y_expansion
        x2 += x2_expansion
        y2 += y2_expansion

        return (x1, y1, x2, y2)

    def resize_pad(self, img, size):
        """
            Resize and pad an image to fit the given size
            Args:
                img (numpy.ndarray): Image to resize
                size (tuple): Size of the new image
                pad_color (int): Color to use for padding
            Returns:
                numpy.ndarray: Resized and padded image
        """
        # Threshold to find non-black regions
        _, thresh = cv2.threshold(img, 1, 255, cv2.THRESH_BINARY)

        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Get bounding box of non-black regions
        x, y, w, h = cv2.boundingRect(contours[0])

        # Crop the image to the bounding box
        cropped_img = img[y:y+h, x:x+w]

        h, w = cropped_img.shape[:2]
        sh, sw = size

        aspect = w/h  

        if aspect > 1: 
            new_w = sw
            new_h = np.round(new_w/aspect).astype(int)
        elif aspect < 1: 
            new_h = sh
            new_w = np.round(new_h*aspect).astype(int)
        else: 
            new_h, new_w = sh, sw

        scaled_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        canvas = np.zeros((size[1], size[0]), dtype=np.uint8)
        offset_x = (size[0] - new_w) // 2
        offset_y = (size[1] - new_h) // 2
        canvas[offset_y:offset_y+new_h, offset_x:offset_x+new_w] = scaled_img

        num_black_pixels = np.sum(canvas == 0)

        # Check if the number of black pixels exceeds the threshold
        if num_black_pixels > 2600:
            return None

        return canvas

    # def get_aligned_face(self, detector, predictor, image):
    #     faces = detector(image)
        
    #     for i, face in enumerate(faces):
    #         landmarks = predictor(image, face)
            
    #         left_eye_points = np.array([(landmarks.part(n).x, landmarks.part(n).y) for n in range(36, 42)], dtype="float32")
    #         right_eye_points = np.array([(landmarks.part(n).x, landmarks.part(n).y) for n in range(42, 48)], dtype="float32")
            
    #         left_eye_center = left_eye_points.mean(axis=0)
    #         right_eye_center = right_eye_points.mean(axis=0)
            
    #         dy = right_eye_center[1] - left_eye_center[1]
    #         dx = right_eye_center[0] - left_eye_center[0]
    #         angle = np.degrees(np.arctan2(dy, dx))  
            
    #         desired_left_eye = (0.35, 0.35)
    #         desired_right_eye = (0.65, 0.35)
            
    #         desired_face_width = 256
    #         desired_face_height = desired_face_width  # Assuming square dimensions for simplicity
    #         desired_dist = desired_right_eye[0] - desired_left_eye[0]
    #         desired_dist *= desired_face_width
    #         scale = desired_dist / np.linalg.norm(right_eye_center - left_eye_center)
            
    #         eyes_center = ((left_eye_center[0] + right_eye_center[0]) / 2,
    #                     (left_eye_center[1] + right_eye_center[1]) / 2)
            
    #         M = cv2.getRotationMatrix2D(tuple(eyes_center), angle, scale)
            
    #         tX = desired_face_width * 0.5
    #         tY = desired_face_height * desired_left_eye[1]
    #         M[0, 2] += (tX - eyes_center[0])
    #         M[1, 2] += (tY - eyes_center[1])
            
    #         (w, h) = (desired_face_width, desired_face_height)
    #         return cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC)

    def align_faces(self, input_dir, output_dir, model_dir):
        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor(str(model_dir))  # Load the face landmark predictor
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        for image_name in os.listdir(input_dir):
            if not (image_name.lower().endswith('.png') or image_name.lower().endswith('.jpg') or image_name.lower().endswith('.jpeg')):
                continue
            
            image_path = os.path.join(input_dir, image_name)
            image = cv2.imread(image_path)
            
            faces = detector(image)
            
            for i, face in enumerate(faces):
                landmarks = predictor(image, face)
                
                left_eye_points = np.array([(landmarks.part(n).x, landmarks.part(n).y) for n in range(36, 42)], dtype="float32")
                right_eye_points = np.array([(landmarks.part(n).x, landmarks.part(n).y) for n in range(42, 48)], dtype="float32")
                
                left_eye_center = left_eye_points.mean(axis=0)
                right_eye_center = right_eye_points.mean(axis=0)
                
                dy = right_eye_center[1] - left_eye_center[1]
                dx = right_eye_center[0] - left_eye_center[0]
                angle = np.degrees(np.arctan2(dy, dx))  
                
                desired_left_eye = (0.35, 0.35)
                desired_right_eye = (0.65, 0.35)
                
                desired_face_width = 256
                desired_face_height = desired_face_width  # Assuming square dimensions for simplicity
                desired_dist = desired_right_eye[0] - desired_left_eye[0]
                desired_dist *= desired_face_width
                scale = desired_dist / np.linalg.norm(right_eye_center - left_eye_center)
                
                eyes_center = ((left_eye_center[0] + right_eye_center[0]) / 2,
                            (left_eye_center[1] + right_eye_center[1]) / 2)
                
                M = cv2.getRotationMatrix2D(tuple(eyes_center), angle, scale)
                
                tX = desired_face_width * 0.5
                tY = desired_face_height * desired_left_eye[1]
                M[0, 2] += (tX - eyes_center[0])
                M[1, 2] += (tY - eyes_center[1])
                
                (w, h) = (desired_face_width, desired_face_height)
                output = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC)
                
                aligned_face_filename = f"{os.path.splitext(image_name)[0]}.jpg"
                aligned_face_path = os.path.join(output_dir, aligned_face_filename)
                cv2.imwrite(aligned_face_path, output)


    