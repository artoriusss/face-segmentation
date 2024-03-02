from pathlib import Path
import shutil
import os 

import numpy as np
import dlib
import facer
import glob
import cv2

class DataHelper():
    def is_frontal_face(shape):
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

    def get_bouding_boxes(rects): 
        """
            Get the bounding boxes of the faces in the image
            Args:
                rects (dlib.rectangles): dlib rectangles object
            Returns:
                tuple: Tuple of bounding boxes
        """
        x1, y1, x2, y2 = (int(value) for value in rects)
        if x1 > 5:
            y1 -= 5
        
        if x2 > 5:
            y2 += 5

        if y1 > 5:
            y1 -= 5
        
        if y2 > 5:
            y2 += 5

        return (x1, y1, x2, y2)

    def resize_pad(img, size, pad_color=0):
        """
            Resize and pad an image to fit the given size
            Args:
                img (numpy.ndarray): Image to resize
                size (tuple): Size of the new image
                pad_color (int): Color to use for padding
            Returns:
                numpy.ndarray: Resized and padded image
        """
        h, w = img.shape[:2]
        sh, sw = size

        interp = cv2.INTER_AREA

        aspect = w/h  

        if aspect > 1: 
            new_w = sw
            new_h = np.round(new_w/aspect).astype(int)
            pad_vert = (sh-new_h)/2
            pad_top, pad_bot = np.floor(pad_vert).astype(int), np.ceil(pad_vert).astype(int)
            pad_left, pad_right = 0, 0
        elif aspect < 1: 
            new_h = sh
            new_w = np.round(new_h*aspect).astype(int)
            pad_horz = (sw-new_w)/2
            pad_left, pad_right = np.floor(pad_horz).astype(int), np.ceil(pad_horz).astype(int)
            pad_top, pad_bot = 0, 0
        else: 
            new_h, new_w = sh, sw
            pad_left, pad_right, pad_top, pad_bot = 0, 0, 0, 0

        padColor = [padColor]*3

        scaled_img = cv2.resize(img, (new_w, new_h), interpolation=interp)
        canvas = np.zeros((size[1], size[0], 3), dtype=np.uint8)
        offset_x = (size[0] - new_w) // 2
        offset_y = (size[1] - new_h) // 2
        canvas[offset_y:offset_y+new_h, offset_x:offset_x+new_w] = scaled_img

        return canvas


    