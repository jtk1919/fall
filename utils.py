import cv2
import json
import pickle
import csv
import numpy as np


def load_json(json_file_path):
    """
    read a json file
    @param json_file_path: (string) path of the json file
    @return: json parsed to a dict
    """
    with open(json_file_path) as file:
        json_dict = json.load(file)
    return json_dict


def imshow_frame(window_name, frame, width, height):
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    # cv2.moveWindow(window_name, 0, 0)
    cv2.resizeWindow(window_name, (width, height))
    cv2.imshow(window_name, frame)

