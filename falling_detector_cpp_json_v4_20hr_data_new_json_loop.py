import os
import time

import cv2
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.cm as cm
from tqdm import tqdm
from scipy.signal import find_peaks

from utils import *


class FrameLooper:

    def __init__(self, video_name, video_root):
        """
        Loops through frame
        @param video_name: (string) name of the video
        @param video_root: (string) folder in which the video is present
        """

        self.video_root = video_root
        self.video_name = video_name
        self.video_path = os.path.join(video_root, self.video_name, self.video_name + '.mp4')


        # read cpp json output
        self.cpp_json_path = os.path.join(self.video_root, self.video_name,
                                          self.video_name + '_v6-202-g8cffb91.json')
        self.cpp_json = None
        try:
            self.cpp_json = load_json(self.cpp_json_path)
        except Exception as e:
            print('could not find the cpp json output in ', self.cpp_json_path)
            print(e)
            pass

        # save cpp json
        self.cpp_json_save_path = os.path.join(self.video_root, self.video_name,
                                          self.video_name + '_v6-202-g8cffb91_output.json')

        # video
        self.video_reader = cv2.VideoCapture(self.video_path)
        self.frame_height = self.video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self.frame_width = self.video_reader.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.src_fps = self.video_reader.get(cv2.CAP_PROP_FPS)  # src fps
        self.frame_index = 0
        self.data_rate = self.cpp_json["outputs"][1]["raw_index"]
        self.dest_fps = self.src_fps / self.data_rate

        # video display utils
        self.pause = False

        self.sticky_size = (192, 256)
        self.sticky_ratio = self.sticky_size[0] / self.sticky_size[1]
        self.sticky_scale = 4.0 / 3.0


        self.track_dict = {}
        self.colormap = cm.get_cmap('Set1')

        self.pose_joint_names = ['nose', 'eye_left', 'eye_right', 'ear_left', 'ear_right', 'shoulder_left',
                                 'shoulder_right', 'elbow_left', 'elbow_right', 'wrist_left', 'wrist_right',
                                 'hip_left', 'hip_right', 'knee_left', 'knee_right', 'ankle_left', 'ankle_right']
        self.viz_dict = {i: joint for i, joint in enumerate(self.pose_joint_names)}
        # all joints
        self.kps_lines = [(1, 2), (0, 1), (0, 2), (2, 4), (1, 3), (6, 8), (8, 10), (5, 7), (7, 9), (12, 14), (14, 16),
                          (11, 13), (13, 15), (5, 6), (11, 12)]
        # trunk, hands and legs
        # self.kps_lines = [(6, 8), (8, 10), (5, 7), (7, 9), (12, 14), (14, 16),
        #                   (11, 13), (13, 15), (5, 6), (11, 12)]
        # trunk and legs
        # self.kps_lines = [(12, 14), (14, 16),
        #                   (11, 13), (13, 15), (5, 6), (11, 12)]
        # trunk and legs
        # self.kps_lines = [(5, 6), (11, 12)]
        self.kp_thresh = 0.4

        # Convert from plt 0-1 RGBA colors to 0-255 BGR colors for opencv.
        cmap = plt.get_cmap('rainbow')
        self.colors = [cmap(i) for i in np.linspace(0, 1, len(self.kps_lines) + 2)]
        self.colors = [(c[2] * 255, c[1] * 255, c[0] * 255) for c in self.colors]


        # define falling detector parameters
        self.vel_delta = 12
        self.var_delta = 25
        self.bending_dilate_delta = 50
        self.CoG_dilate_delta = 25
        self.delta_min = 5
        self.conf_delta = 12
        self.smooth_window = 3

        # a function to dilate the bending and detection
        # if there are more than certain frames of 1 in a rolling window, then set the current value to 1 as well
        self.dilate_bending = lambda x: (1 if np.sum(x == 1) >= 2 else x.iloc[-1])
        self.dilate_var_CoG = lambda x:(1 if np.sum(x==1) >= 1 else x.iloc[-1])

        self.interesting_trackids = [3, 12, 57, 488, 456, 932, 1122, 1186, 1196, 1202, 1210, 1637, 1666, 1716, 1736, 1823, 1828, 2135, 2275, 2944, 3217, 3226]
        self.start_frame_index = 76794 # 26140, 52703-30, 59737, 76794
        self.plt_frame_index = [201, 474, 1184, 26020, 26062, 26202, 26253, 52738, 59799, 76912]
        self.plot = True

    def draw_stickies(self, key_points, frame):
        if len(key_points) == 0:
            return frame
        # Draw mid shoulder / mid hip first for better visualization.
        mid_shoulder = (key_points[5, :2] + key_points[6, :2]) / 2.0
        sc_mid_shoulder = np.minimum(key_points[5, 2], key_points[6, 2])
        mid_hip = (key_points[11, :2] + key_points[12, :2]) / 2.0
        sc_mid_hip = np.minimum(key_points[11, 2], key_points[12, 2])
        nose_idx = 0
        if sc_mid_shoulder > self.kp_thresh and key_points[nose_idx, 2] > self.kp_thresh:
            cv2.line(
                frame, tuple(mid_shoulder.astype(np.int32)), tuple(key_points[nose_idx, :2].astype(np.int32)),
                color=self.colors[len(self.kps_lines)], thickness=2, lineType=cv2.LINE_AA)
        if sc_mid_shoulder > self.kp_thresh and sc_mid_hip > self.kp_thresh:
            cv2.line(
                frame, tuple(mid_shoulder.astype(np.int32)), tuple(mid_hip.astype(np.int32)),
                color=self.colors[len(self.kps_lines) + 1], thickness=2, lineType=cv2.LINE_AA)

        # Draw the keypoints.
        for l in range(len(self.kps_lines)):
            i1 = self.kps_lines[l][0]
            i2 = self.kps_lines[l][1]
            p1 = key_points[i1, 0].astype(np.int32), key_points[i1, 1].astype(np.int32)
            p2 = key_points[i2, 0].astype(np.int32), key_points[i2, 1].astype(np.int32)
            if key_points[i1, 2] > self.kp_thresh and key_points[i2, 2] > self.kp_thresh:
                cv2.line(
                    frame, p1, p2,
                    color=self.colors[l], thickness=2, lineType=cv2.LINE_AA)
            if key_points[i1, 2] > self.kp_thresh:
                cv2.circle(
                    frame, p1,
                    radius=3, color=self.colors[l], thickness=-1, lineType=cv2.LINE_AA)
            if key_points[i2, 2] > self.kp_thresh:
                cv2.circle(
                    frame, p2,
                    radius=3, color=self.colors[l], thickness=-1, lineType=cv2.LINE_AA)
        return frame

    def convert_point_from_sticky2bbox(self, x, y, bbox_width, bbox_height):
        # x and y are in sticky size coord. Convert them to bbox
        sticky_width = self.sticky_size[0]
        sticky_height = self.sticky_size[1]

        # convert the point from wrt sticky to wrt bbox
        new_x = int(bbox_width * (x / sticky_width))
        new_y = int(bbox_height * (y / sticky_height))

        return new_x, new_y

    def convert_point_from_bbox2enlargedbbox(self, x, y, bbox_width, bbox_height):

        enlarged_height = self.sticky_size[1]
        enlarged_width = enlarged_height * (bbox_width/bbox_height)

        # convert the point from wrt sticky to wrt bbox
        new_x = int(enlarged_width * (x / bbox_width))
        new_y = int(enlarged_height * (y / bbox_height))

        return new_x, new_y

    def convert_point_from_bbox2sticky(self, x, y, bbox_width, bbox_height):
        # x and y are in bbox coord. Convert them to sticky size
        sticky_width = self.sticky_size[0]
        sticky_height = self.sticky_size[1]

        # convert the point from wrt bbox to wrt sticky
        new_x = int(sticky_width * (x / bbox_width))
        new_y = int(sticky_height * (y / bbox_height))

        return new_x, new_y

    def rescale_bbox_for_stickies(self, bbox):
        # rescales bbox for drawing stickies and rescaling stickies to bbox coord
        bbox_width = bbox['right'] - bbox['left']
        bbox_height = bbox['bottom'] - bbox['top']

        cx = (bbox['left'] + bbox['right']) / 2
        cy = (bbox['top'] + bbox['bottom']) / 2

        if (bbox_width > bbox_height * self.sticky_ratio):
            bbox_height = bbox_width / self.sticky_ratio
        elif (bbox_width < bbox_height * self.sticky_ratio):
            bbox_width = bbox_height * self.sticky_ratio

        bbox_width = np.floor(bbox_width * self.sticky_scale)
        bbox_height = np.floor(bbox_height * self.sticky_scale)
        bbox['left'] = (cx - bbox_width / 2)
        bbox['top'] = (cy - bbox_height / 2)
        bbox['right'] = (cx + bbox_width / 2)
        bbox['bottom'] = (cy + bbox_height / 2)

        return bbox

    def person_person_occlusion(self, current_det, all_detections):
        # current bbox
        bbox = current_det['norm_bounding_box']
        current_crunch = [int(bbox['left']), int(bbox['top']), int(bbox['right']), int(bbox['bottom'])]

        # gather all the candidate bboxes
        candidates = []
        for query_det in all_detections:
            if query_det['id'] != current_det['id']:
                bbox = query_det['norm_bounding_box']
                query_det_crunch = [int(bbox['left']), int(bbox['top']), int(bbox['right']), int(bbox['bottom'])]
                candidates.append(query_det_crunch)

        if len(candidates) > 0:
            candidates = np.array(candidates)
            current_crunch = np.array(current_crunch)
            # find intersection over union between current bbox and all candidates
            iou = get_occ_iou(current_crunch, candidates)
            if iou> 0.01:
                occlusion = 1
            else:
                occlusion = 0
        else:
            occlusion = 0
        return occlusion

    def run(self):
        """
        Loops through the cpp json frame by frame, performs falling detection and populates/saves cpp json with results
        """

        total_time = 0
        total_frame = len(self.cpp_json['outputs'])

        for self.frame_index in range(len(self.cpp_json["outputs"])):
            start = time.time()
            self.raw_frame_index = self.cpp_json["outputs"][self.frame_index]['raw_index']

            # do stuff here
            if self.frame_index % 10000 == 0: print(f"{self.frame_index} / {total_frame}")
            detections = self.cpp_json["outputs"][self.frame_index]["detections"]
            detections = [det for det in detections if 'id' in det.keys() and 'body_skeleton' in det.keys()
                          and len(det['body_skeleton']) > 0]
            for ii in range(len(detections)):
                det = detections[ii]

                # create track id for track analysis
                if det['id'] not in self.track_dict:
                    self.track_dict[det['id']] = {}
                frame_dict, key_points = self.fill_frame_dict(self.frame_index, det)
                pp_occ = self.person_person_occlusion(det, detections)
                frame_dict['occlusion'] = frame_dict['occlusion'] | pp_occ
                self.track_dict[det['id']][self.frame_index] = frame_dict
                # only have the latest frame data and remove the rest
                # self.track_dict[det['id']] = self.track_dict[det['id']][-int(3*self.src_fps):]
                self.track_dict[det['id']] = {key: value for key, value in self.track_dict[det['id']].items()
                                              if key in range(self.frame_index - int(75),
                                                              self.frame_index + 1)}

                # detect falling
                detection_conf, bending_conf = self.falling_detector(self.track_dict[det['id']], det['id'])
                if detection_conf > 0:
                    print(self.frame_index, det['id'], 'det', np.round(detection_conf, 2))
                if bending_conf > 0:
                    print(self.frame_index, det['id'], 'bending', np.round(bending_conf, 2))
                # populate cpp json
                self.cpp_json["outputs"][self.frame_index]["detections"][ii]['falling_detection_conf'] = detection_conf
                self.cpp_json["outputs"][self.frame_index]["detections"][ii]['bending_detection_conf'] = bending_conf
                self.cpp_json["outputs"][self.frame_index]["detections"][ii]['occlusion'] = frame_dict['occlusion']
                duration = time.time() - start
                total_time = total_time + duration
                # print('Time taken ', duration)

        # save cpp json
        save_json(self.cpp_json_save_path, self.cpp_json)
        print('Total time ', total_time)



    def fill_frame_dict(self, frame_index, det):
        frame_dict = {'trackid': det['id'], 'frame_index': frame_index}

        # bbox
        bbox = det['norm_bounding_box']
        frame_dict['bb_left'], frame_dict['bb_top'], frame_dict['bb_right'], frame_dict['bb_bottom'] = \
            int(bbox['left']), int(bbox['top']), int(bbox['right']), int(bbox['bottom'])

        bbox_rescaled = self.rescale_bbox_for_stickies(bbox.copy())

        bbox_width_rescaled = bbox_rescaled['right'] - bbox_rescaled['left']
        bbox_height_rescaled = bbox_rescaled['bottom'] - bbox_rescaled['top']
        bbox_width = bbox['right'] - bbox['left']
        bbox_height = bbox['bottom'] - bbox['top']

        # pose
        pose = det['body_skeleton']
        key_points_bbox = []
        key_points_actual_bbox = []
        key_points_frame = []
        key_points_norm = []
        for key, value in pose.items():
            # keypoint in frame coord
            x, y = value['x'], value['y']
            key_points_frame.append([x, y, value['score']])
            frame_dict[key + '_frame_x'] = x
            frame_dict[key + '_frame_y'] = y
            frame_dict[key + '_frame_score'] = value['score']

            # convert keypoint from frame to bbox rescaled (to center the sticky and add borders)
            x = value['x'] - bbox_rescaled['left']
            y = value['y'] - bbox_rescaled['top']
            # convert keypoint from bbox rescaled size to sticky size, i.e.
            # normalised keypoint with respect to sticky size
            x, y = self.convert_point_from_bbox2sticky(x, y, bbox_width_rescaled, bbox_height_rescaled)
            key_points_norm.append([x, y, value['score']])
            frame_dict[key + '_x'] = x
            frame_dict[key + '_y'] = y
            frame_dict[key + '_score'] = value['score']

            # convert keypoint from sticky size to bbox size
            x_bbox, y_bbox = self.convert_point_from_sticky2bbox(x, y, bbox_width, bbox_height)
            key_points_bbox.append([x_bbox, y_bbox, value['score']])
            frame_dict[key + '_bbox_x'] = x_bbox
            frame_dict[key + '_bbox_y'] = y_bbox
            frame_dict[key + '_bbox_score'] = value['score']

            # convert keypoint from frame to actual bbox size
            x_actual_bbox = value['x'] - bbox['left']
            y_actual_bbox = value['y'] - bbox['top']
            # convert keypoint from actual bbox to enlarged/standard height bbox
            x_actual_bbox, y_actual_bbox = self.convert_point_from_bbox2enlargedbbox(x_actual_bbox, y_actual_bbox,
                                                                                     bbox_width, bbox_height)
            key_points_actual_bbox.append([x_actual_bbox, y_actual_bbox, value['score']])

            frame_dict[key + '_actual_bbox_x'] = x_actual_bbox
            frame_dict[key + '_actual_bbox_y'] = y_actual_bbox
            frame_dict[key + '_actual_bbox_score'] = value['score']

        # find CoG
        key_points_bbox = np.array(key_points_bbox)
        CoG_bbox = self.find_CoG_from_pose(key_points_bbox)
        frame_dict['CoG_bbox_x'] = CoG_bbox[0]
        frame_dict['CoG_bbox_y'] = CoG_bbox[1]
        frame_dict['CoG_bbox_score'] = CoG_bbox[2]
        key_points_frame = np.array(key_points_frame)
        CoG_frame = self.find_CoG_from_pose(key_points_frame)
        frame_dict['CoG_frame_x'] = CoG_frame[0]
        frame_dict['CoG_frame_y'] = CoG_frame[1]
        frame_dict['CoG_frame_score'] = CoG_frame[2]
        key_points_norm = np.array(key_points_norm)
        CoG_norm = self.find_CoG_from_pose(key_points_norm)
        frame_dict['CoG_norm_x'] = CoG_norm[0]
        frame_dict['CoG_norm_y'] = CoG_norm[1]
        frame_dict['CoG_norm_score'] = CoG_norm[2]

        # occluded zone
        # if 85 < bbox['left'] < 500 and bbox['bottom'] < self.frame_height - 10:
        # if 10 < bbox['left'] and bbox['right'] < self.frame_width-10 and 200 < bbox['bottom'] < self.frame_height - 10:
        #     occlusion = 0
        # else:
        #     occlusion = 1
        occlusion = 0
        frame_dict['occlusion'] = occlusion


        return frame_dict, key_points_norm


    def falling_detector(self, track_framedict_list, trackid):

        track_info = pd.DataFrame.from_dict(track_framedict_list, orient='index')
        track_info = track_info.reindex(list(range(track_info.index.min(), track_info.index.max() + 1)), fill_value=np.nan)

        # position of bbox in image based occlusion
        occlusion_cond = track_info['occlusion'] == 1

        # bending detection
        # find shoulder, hip, ankle middle
        track_info['shoulder_middle_x'] = (track_info['shoulder_left_x'] + track_info['shoulder_right_x']) / 2
        track_info['shoulder_middle_y'] = (track_info['shoulder_left_y'] + track_info['shoulder_right_y']) / 2
        track_info['shoulder_middle_score'] = np.mean([track_info['shoulder_left_score'],
                                                         track_info['shoulder_right_score']], axis=0)

        track_info['hip_middle_x'] = (track_info['hip_left_x'] + track_info['hip_right_x']) / 2
        track_info['hip_middle_y'] = (track_info['hip_left_y'] + track_info['hip_right_y']) / 2
        track_info['hip_middle_score'] = np.mean([track_info['hip_left_score'], track_info['hip_right_score']], axis=0)

        # distance between shoulder and hip
        track_info['dist_shoulder_hip_y'] = track_info['shoulder_middle_y'] - track_info['hip_middle_y']
        track_info['dist_shoulder_hip_score'] = np.mean([track_info['shoulder_middle_score'],
                                                           track_info['hip_middle_score']], axis=0)
        # some filtering
        track_info.loc[(track_info['dist_shoulder_hip_score'] < self.kp_thresh) | (occlusion_cond),
                       ['dist_shoulder_hip_y']] = np.nan
        # some smoothing
        track_info['dist_shoulder_hip_y'] = track_info['dist_shoulder_hip_y'].rolling(
            self.smooth_window).mean()
        if trackid in self.interesting_trackids and self.plot and self.frame_index in self.plt_frame_index:
            self.plot_function(track_info, 'dist_shoulder_hip', trackid)

        # find trunk angle
        track_info['trunk_angle_y'] = np.rad2deg(
            np.arctan2(track_info['shoulder_middle_y'] - track_info['hip_middle_y'],
                       track_info['shoulder_middle_x'] - track_info['hip_middle_x']))
        track_info['trunk_angle_score'] = np.mean([track_info['shoulder_middle_score'],
                                                     track_info['hip_middle_score']], axis=0)
        # some filtering
        track_info.loc[(track_info['trunk_angle_score'] < self.kp_thresh) | (occlusion_cond),
                       ['trunk_angle_y']] = np.nan
        # some smoothing
        track_info['trunk_angle_y'] = track_info['trunk_angle_y'].rolling(self.smooth_window).mean()
        if trackid in self.interesting_trackids and self.plot and self.frame_index in self.plt_frame_index:
            self.plot_function(track_info, 'trunk_angle', trackid)


        # # # bending
        track_info['bending'] = np.zeros_like(track_info['occlusion'])
        trunk_angle_condition = ((track_info['trunk_angle_y'] > -70) | (track_info['trunk_angle_y'] < -120))
        bending_condition = ((track_info['dist_shoulder_hip_y'] > -44) & trunk_angle_condition) | trunk_angle_condition

        track_info.loc[bending_condition, ['bending']] = 1

        # dilate bending
        track_info['bending'] = track_info['bending'].rolling(self.bending_dilate_delta,
                                                              min_periods=self.delta_min).apply(self.dilate_bending)

        # # # legs occluded
        ankle_left_score_low_condition = (track_info['ankle_left_score'] < self.kp_thresh)
        ankle_right_score_low_condition = (track_info['ankle_right_score'] < self.kp_thresh)

        track_info['ankle_max_actual_bbox_y'] = np.nan * np.ones_like(track_info['ankle_left_actual_bbox_y'])
        track_info.loc[
            ~ ankle_left_score_low_condition & ~ ankle_right_score_low_condition, ['ankle_max_actual_bbox_y']] = \
            np.maximum(track_info['ankle_left_actual_bbox_y'], track_info['ankle_right_actual_bbox_y'])
        track_info.loc[
            ~ ankle_left_score_low_condition & ankle_right_score_low_condition, ['ankle_max_actual_bbox_y']] = \
            track_info['ankle_left_actual_bbox_y']
        track_info.loc[
            ankle_left_score_low_condition & ~ ankle_right_score_low_condition, ['ankle_max_actual_bbox_y']] = \
            track_info['ankle_right_actual_bbox_y']

        # enlarged bbox height
        enlarged_bbox_height = self.sticky_size[1]
        # difference between the ankle and bbox bottom line in y direction
        track_info['ankle_from_actual_bbox_bottom_y'] = track_info['ankle_max_actual_bbox_y'] - enlarged_bbox_height
        track_info['ankle_from_actual_bbox_bottom_score'] = np.maximum(track_info['ankle_left_score'],
                                                                       track_info['ankle_right_score'])

        # # variability
        # track_info['var_ankle_from_actual_bbox_bottom_y'] = track_info['ankle_from_actual_bbox_bottom_y'].rolling(
        #     self.var_delta, min_periods=self.delta_min).std()
        # track_info['var_ankle_from_actual_bbox_bottom_score'] = track_info['ankle_from_actual_bbox_bottom_score']
        # if trackid in self.interesting_trackids and self.plot and self.frame_index in self.plt_frame_index:
        #     self.plot_function(track_info, 'ankle_from_actual_bbox_bottom', trackid)
        #     # self.plot_function(track_info, 'var_ankle_from_actual_bbox_bottom', trackid)

        # legs based occlusion
        track_info.loc[ankle_left_score_low_condition & ankle_right_score_low_condition, ['occlusion']] = 1
        track_info.loc[(track_info['ankle_from_actual_bbox_bottom_y'] > 15), ['occlusion']] = 1

        occlusion_cond = track_info['occlusion'] == 1


        # # # trip fall detection
        # CoG
        # some filtering
        track_info.loc[(track_info['CoG_norm_score'] < self.kp_thresh) | (occlusion_cond), ['CoG_norm_x','CoG_norm_y']] = np.nan
        # smoothing
        track_info['smooth_CoG_norm_x'] = track_info['CoG_norm_x'].rolling(self.smooth_window).mean()
        track_info['smooth_CoG_norm_y'] = track_info['CoG_norm_y'].rolling(self.smooth_window).mean()
        track_info['smooth_CoG_norm_score'] = track_info['CoG_norm_score']

        # variability of CoG
        track_info['var_smooth_CoG_norm_x'] = track_info['smooth_CoG_norm_x'].rolling(self.var_delta, min_periods=self.delta_min).std()
        track_info['var_smooth_CoG_norm_y'] = track_info['smooth_CoG_norm_y'].rolling(self.var_delta, min_periods=self.delta_min).std()
        track_info['var_smooth_CoG_norm_score'] = track_info['smooth_CoG_norm_score']
        if trackid in self.interesting_trackids and self.plot and self.frame_index in self.plt_frame_index:
            self.plot_function(track_info, 'var_smooth_CoG_norm', trackid)


        # velocity of CoG
        track_info['vel_CoG_norm_x'] = track_info['smooth_CoG_norm_x'].diff(periods=self.vel_delta)
        track_info['vel_CoG_norm_y'] = track_info['smooth_CoG_norm_y'].diff(periods=self.vel_delta)
        track_info['vel_CoG_norm_score'] = track_info['smooth_CoG_norm_score']
        if trackid in self.interesting_trackids and self.plot and self.frame_index in self.plt_frame_index:
            self.plot_function(track_info, 'vel_CoG_norm', trackid)


        ### CoG_bbox of legs
        track_info['hip_middle_bbox_x'] = (track_info['hip_left_bbox_x'] + track_info['hip_right_bbox_x']) / 2
        track_info['hip_middle_bbox_y'] = (track_info['hip_left_bbox_y'] + track_info['hip_right_bbox_y']) / 2
        track_info['hip_middle_bbox_score'] = np.mean([track_info['hip_left_bbox_score'],
                                                         track_info['hip_right_bbox_score']], axis=0)
        dist_x, dist_y, score = [], [], []
        for joint in ['ankle_left', 'ankle_right', 'knee_left', 'knee_right', 'hip_left', 'hip_right']:
            track_info['dist_' + joint + '_x'] = track_info[joint + '_bbox_x'] - track_info['hip_middle_bbox_x']
            track_info['dist_' + joint + '_y'] = track_info[joint + '_bbox_y'] - track_info['hip_middle_bbox_y']
            track_info['dist_' + joint + '_score'] = np.minimum(track_info[joint + '_bbox_score'],
                                                                track_info['hip_middle_bbox_score'])

            dist_x.append(track_info['dist_' + joint + '_x'])
            dist_y.append(track_info['dist_' + joint + '_y'])
            score.append(track_info[joint + '_score'])

        track_info['CoG_bbox_legs_x'] = np.mean(dist_x, axis=0)
        track_info['CoG_bbox_legs_y'] = np.mean(dist_y, axis=0)
        track_info['CoG_bbox_legs_score'] = np.mean(score, axis=0)
        # filtering
        track_info.loc[(track_info['CoG_bbox_legs_score'] < self.kp_thresh) | (occlusion_cond),
                       ['CoG_bbox_legs_x', 'CoG_bbox_legs_y']] = np.nan
        # smoothing
        track_info['CoG_bbox_legs_x'] = track_info['CoG_bbox_legs_x'].rolling(self.smooth_window).mean()
        track_info['CoG_bbox_legs_y'] = track_info['CoG_bbox_legs_y'].rolling(self.smooth_window).mean()

        # variability
        track_info['var_CoG_bbox_legs_x'] = track_info['CoG_bbox_legs_x'].rolling(self.var_delta,
                                                                                  min_periods=self.delta_min).std()
        track_info['var_CoG_bbox_legs_y'] = track_info['CoG_bbox_legs_y'].rolling(self.var_delta,
                                                                                  min_periods=self.delta_min).std()
        track_info['var_CoG_bbox_legs_score'] = track_info['CoG_bbox_legs_score']
        if trackid in self.interesting_trackids and self.plot and self.frame_index in self.plt_frame_index:
            self.plot_function(track_info, 'var_CoG_bbox_legs', trackid)


        # condition on var_CoG_bbox_legs
        track_info['var_CoG_bbox_legs'] = np.zeros_like(track_info['occlusion'])
        track_info.loc[(np.abs(track_info['var_CoG_bbox_legs_y']) >= 2), ['var_CoG_bbox_legs']] = 1

        # detection
        track_info['detection'] = np.zeros_like(track_info['occlusion'])
        detection_condition = (np.abs(track_info['var_smooth_CoG_norm_y']) > 4) | \
                              (np.abs(track_info['vel_CoG_norm_y']) > 8) | \
                              (np.abs(track_info['var_CoG_bbox_legs_y']) > 4)
        track_info.loc[detection_condition, ['detection']] = 1


        # dilate var_CoG_bbox_legs
        track_info['var_CoG_bbox_legs'] = track_info['var_CoG_bbox_legs'].rolling(self.CoG_dilate_delta,
                                                                   min_periods=self.delta_min).apply(self.dilate_var_CoG)

        # remove false positives from detection using dilated var_CoG_bbox_legs
        track_info.loc[(np.abs(track_info['var_CoG_bbox_legs']) == 0), ['detection']] = 0


        # remove bending from falling detection
        track_info.loc[track_info['bending'] == 1, ['detection']] = 0



        if trackid in self.interesting_trackids and self.plot and self.frame_index in self.plt_frame_index:
            self.plot_function(track_info, 'smooth_CoG_norm', trackid, None, color_label='detection')
            self.plot_function(track_info, 'smooth_CoG_norm', trackid, None, color_label='bending')
            plt.show()


        # detection confidence
        # track_info['detection_conf'] = track_info['detection'].rolling(conf_delta, min_periods=var_delta_min).sum()/conf_delta
        limit = np.min([len(track_info['detection']), self.conf_delta])
        detection_conf = np.sum(track_info['detection'].iloc[-limit:]) / self.conf_delta
        if np.isnan(detection_conf): detection_conf = 0
        detection_conf = np.round(detection_conf, 2)

        # bending confidence
        # track_info['bending_conf'] = track_info['bending'].rolling(conf_delta, min_periods=var_delta_min).sum()/conf_delta
        limit = np.min([len(track_info['bending']), self.conf_delta])
        bending_conf = np.sum(track_info['bending'].iloc[-limit:]) / self.conf_delta
        if np.isnan(bending_conf): bending_conf = 0
        bending_conf = np.round(bending_conf, 2)

        # one more chck to remove bending false positives
        if bending_conf > 0:
            detection_conf = 0

        return detection_conf, bending_conf

    def find_CoG_from_pose(self, keypoints, all_joints=False):

        if all_joints:
            start_index = 0
        else:
            start_index = 5

        if len(keypoints) == 0:
            return [np.nan, np.nan, np.nan]

        hip_middle = [int((keypoints[11, 0] + keypoints[12, 0]) / 2), int((keypoints[11, 1] + keypoints[12, 1]) / 2)]
        dist_x, dist_y = [], []
        for point in keypoints[start_index:, :]:
            dist_x.append(point[0] - hip_middle[0])
            dist_y.append(point[1] - hip_middle[1])
        mean_score = np.mean(keypoints[start_index:, 2])
        # CoG = [int(np.mean(dist_x) + hip_middle[0]), int(np.mean(dist_y) + hip_middle[1]), mean_score]
        CoG = [int(np.mean(dist_x)), int(np.mean(dist_y)), mean_score]

        return CoG

    def find_CoG_from_legs(self, keypoints):

        if len(keypoints)==0:
            return [np.nan,np.nan,np.nan]

        # count = len([elem for elem in keypoints[5:6,2] + keypoints[11:12,2] if elem < self.kp_thresh])
        # if count>0:
        #     return [np.nan,np.nan,np.nan]

        hip_middle = [int((keypoints[11,0]+keypoints[12,0])/2), int((keypoints[11,1]+keypoints[12,1])/2)]
        dist_x, dist_y = [], []
        for point in keypoints[11:,:]:
            dist_x.append(point[0]-hip_middle[0])
            dist_y.append(point[1]-hip_middle[1])
        mean_score = np.mean(keypoints[5:6,2] + keypoints[11:12,2])
        # CoG = [int(np.mean(dist_x) + hip_middle[0]), int(np.mean(dist_y) + hip_middle[1]), mean_score]
        CoG = [int(np.mean(dist_x)), int(np.mean(dist_y)), mean_score]

        return CoG


    def plot_function(self, track_info, joint_name, trackid, root_dir=None, color_label='None'):

        fig, ax = plt.subplots(3)
        fig.suptitle(color_label + ' ' + joint_name + ' trackid-' + str(trackid))
        ax[0].set(xlabel='frame_index', ylabel='xpos')
        ax[1].set(xlabel='frame_index', ylabel='ypos')
        ax[2].set(xlabel='frame_index', ylabel='score')

        color = np.ones(shape=(len(track_info['frame_index']), 3))
        color[:] = (0, 0, 1)
        if color_label!= 'None':
            color[track_info[color_label] == 1] = (1, 0, 0)

        # # filtering based on scores
        # color = color[track_info[joint_name+'_score']>0.6]
        # track_info_clean = track_info[track_info[joint_name+'_score']>0.6]

        if joint_name + '_x' in track_info.keys():
            ax[0].scatter(track_info['frame_index'], track_info[joint_name + '_x'], c=color)
        ax[1].scatter(track_info['frame_index'], track_info[joint_name + '_y'], c=color)
        ax[2].scatter(track_info['frame_index'], track_info[joint_name + '_score'], c=color)

        # frame index limits
        limit = track_info['frame_index'].iloc[-1] - track_info['frame_index'].iloc[0]
        left = track_info['frame_index'].iloc[0]
        if limit < 500:
            right = track_info['frame_index'].iloc[-1] + (500 - limit)
        else:
            right = track_info['frame_index'].iloc[-1]

        ax[0].set_xlim([left, right])
        ax[1].set_xlim([left, right])
        ax[2].set_xlim([left, right])

        # ax[0].set_ylim([-25, 25])
        # ax[1].set_ylim([-25, 25])
        # ax[2].set_ylim([0, 1])

        # plt.savefig(os.path.join(root_dir,color_label+ ' '+joint_name+' trackid-' + str(trackid)+'.png'))


if __name__ == "__main__":
    video_root = '/home/yazhini/dB/shared_videos/Kajima/20_hr/20_hr_camera2'

    t0 = time.time()
    for video_name in os.listdir(video_root):
        t1 = time.time()
        print(video_name, ' #####################################')
        frame_looper = FrameLooper(video_name, video_root)
        frame_looper.run()
        t2 = time.time()
        print(f"Time: {t2 - t1} | TOTAL: {t2 - t0}")