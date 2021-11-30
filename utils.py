import json
import numpy as np


def save_json(json_file_path, json_data):
    """
    save a json file
    @param json_file_path: (string) path of the json file
    @return: data to be saved as json
    """
    with open(json_file_path, 'w') as f:
        json.dump(json_data, f)


def load_json(json_file_path):
    """
    read a json file
    @param json_file_path: (string) path of the json file
    @return: json parsed to a dict
    """
    with open(json_file_path) as file:
        json_dict = json.load(file)
    return json_dict


def get_occ_iou(bbox, candidates):
    """Computer intersection over union.
    Parameters
    ----------
    bbox : ltrb, crunch
    candidates : ndarray
        A matrix of candidate bounding boxes (one per row) in the same format
        as `bbox`.
    Returns
    -------
    ndarray
        The intersection over union in [0, 1] between the `bbox` and each
        candidate. A higher score means a larger fraction of the `bbox` is
        occluded by the candidate.
    """
    bbox_tl, bbox_br = bbox[:2], bbox[2:]
    bbox_wh = bbox_br - bbox_tl
    candidates_tl = candidates[:, :2]
    candidates_br = candidates[:, 2:]
    candidates_wh = candidates_br - candidates_tl

    tl = np.c_[np.maximum(bbox_tl[0], candidates_tl[:, 0])[:, np.newaxis],
               np.maximum(bbox_tl[1], candidates_tl[:, 1])[:, np.newaxis]]
    br = np.c_[np.minimum(bbox_br[0], candidates_br[:, 0])[:, np.newaxis],
               np.minimum(bbox_br[1], candidates_br[:, 1])[:, np.newaxis]]
    wh = np.maximum(0., br - tl)

    area_intersection = wh.prod(axis=1)
    area_bbox = bbox_wh.prod()
    area_candidates = candidates_wh.prod(axis=1)
    occ_iou_value = area_intersection / (area_bbox)
    # iou_value = area_intersection / (area_bbox + area_candidates - area_intersection)
    occ_iou_value = np.max(occ_iou_value)
    return occ_iou_value