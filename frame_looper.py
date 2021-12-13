import os
import time
import cv2
from .utils import *


class FrameLooper:

    def __init__(self, video_name, video_root):
        """
        Loops through cpp json output data, performs falling detection on the data and updates the cpp json with
        falling detection. The following are added to the cpp json output. For each detection in a frame,
        falling_detection_confidence (float- 0 to 1), bending_detection_confidence (float- 0 to 1), occlusion (bool)

        @param video_name: (string) name of the video
        @param video_root: (string) folder in which the video is present
        """

        self.video_root = video_root
        self.video_name = video_name
        self.video_path = os.path.join(video_root, self.video_name, self.video_name + '.mp4')
        # check if the video exists at the given path
        if os.path.exists(self.video_path):
            self.video_found = True
        else:
            self.video_found = False
            print('Error: Could not find the video file in ', self.video_path)

        # read cpp json output
        self.cpp_json_path = os.path.join(self.video_root, self.video_name,
                                          self.video_name + '.json')
        try:
            self.cpp_json = load_json(self.cpp_json_path)
            self.cpp_json_found = True
        except Exception as e:
            self.cpp_json = None
            self.cpp_json_found = False
            print(e)
            pass

        # save cpp json
        self.cpp_json_save_path = os.path.join(self.video_root, self.video_name,
                                               self.video_name + '_output.json')

        # read video and get the video settings
        self.video_reader = cv2.VideoCapture(self.video_path)
        self.frame_height = self.video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self.frame_width = self.video_reader.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.src_fps = self.video_reader.get(cv2.CAP_PROP_FPS)  # src fps
        self.frame_index = 0
        self.data_rate = self.cpp_json["outputs"][1]["raw_index"] if self.cpp_json is not None else None
        self.dest_fps = self.src_fps / self.data_rate if self.data_rate is not None else None

        # pose sticky settings
        self.sticky_size = (192, 256)  # sticky model input size (width, height)
        self.sticky_ratio = self.sticky_size[0] / self.sticky_size[1]
        self.sticky_scale = 4.0 / 3.0

        # list of pose joint names for reference
        # pose_joint_names = ['nose', 'eye_left', 'eye_right', 'ear_left', 'ear_right', 'shoulder_left',
        #                          'shoulder_right', 'elbow_left', 'elbow_right', 'wrist_left', 'wrist_right',
        #                          'hip_left', 'hip_right', 'knee_left', 'knee_right', 'ankle_left', 'ankle_right']

        self.kp_thresh = 0.4  # score threshold for a pose joint to be noise free

        self.track_dict = {}  # dict to store track data

        # define falling detector parameters
        self.vel_delta = 12
        self.var_delta = 25
        self.bending_dilate_delta = 50
        self.CoG_dilate_delta = 25
        self.delta_min = 5
        self.conf_delta = 12
        self.smooth_window = 3

        # a function to dilate the bending and detection
        # in a window, if the sum exceeds a threshold, then return 1 else return the same value as before
        self.dilate_bending = lambda x: (1 if np.nansum(x == 1) >= 2 else x[-1])
        self.dilate_var_CoG = lambda x: (1 if np.nansum(x == 1) >= 1 else x[-1])

        self.person_person_occ_thresh = 0.01  # threshold for iou based occlusion detection

    def convert_point_from_sticky2bbox(self, x, y, bbox_width, bbox_height):
        """
        Converts an x,y pixel value from with respect to sticky height, width to bbox height, width
        Example a pose joint has x,y pixel value and is defined wrt sticky height, width. We want the pose joint
        wrt bbox height,width. Then use this function.

        :param x: (int) horizontal x value of pixel in image
        :param y: (int) vertical y value of pixel in image
        :param bbox_width: (int) width of the bbox
        :param bbox_height: (int) height of the bbox
        :return: converted x,y pixel value, (int), (int)
        """
        sticky_width = self.sticky_size[0]
        sticky_height = self.sticky_size[1]

        # convert the point from wrt sticky to wrt bbox
        new_x = int(bbox_width * (x / sticky_width))
        new_y = int(bbox_height * (y / sticky_height))

        return new_x, new_y

    def convert_point_from_bbox2sticky(self, x, y, bbox_width, bbox_height):
        """
        Converts an x,y pixel value from with respect to bbox height, width to sticky height, width.

        :param x: (int) horizontal x value of pixel in image
        :param y: (int) vertical y value of pixel in image
        :param bbox_width: (int) width of the bbox
        :param bbox_height: (int) height of the bbox
        :return: converted x,y pixel value, (int), (int)
        """
        sticky_width = self.sticky_size[0]
        sticky_height = self.sticky_size[1]

        # convert the point from wrt bbox to wrt sticky
        new_x = int(sticky_width * (x / bbox_width))
        new_y = int(sticky_height * (y / bbox_height))

        return new_x, new_y

    def convert_point_from_bbox2enlargedbbox(self, x, y, bbox_width, bbox_height):
        """
        Converts an x,y pixel value from with respect to bbox height,width to a
        fixed height and same aspect ratio as bbox
        Example a pose joint has x,y pixel value and is defined wrt bbox height, width.
        We enlarge the bbox height respecting its original aspect ratio.
        Then we find the pose joint wrt to this enlarged bbox.

        :param x: (int) horizontal x value of pixel in image
        :param y: (int) vertical y value of pixel in image
        :param bbox_width: (int) width of the bbox
        :param bbox_height: (int) height of the bbox
        :return: converted x,y pixel value, (int), (int)
        """
        enlarged_height = self.sticky_size[1]  # fixed height that we want
        enlarged_width = enlarged_height * (bbox_width / bbox_height)  # corresponding width respecting
        # aspect ratio of bbox

        # convert the point from wrt bbox to wrt enlarged bbox
        new_x = int(enlarged_width * (x / bbox_width))
        new_y = int(enlarged_height * (y / bbox_height))

        return new_x, new_y

    def rescale_bbox_for_stickies(self, bbox):
        """
        Rescales the bbox left, top, right, bottom to make the stickies centered. That is,
        adding some padding around stickies so that the stickies look centered wrt bbox. This is because,
        sticky model gets this type of padded bbox as input, which we call a viddie.
        This function converts bbox to viddie.

        :param bbox: (dict) bbox with top, left, right, bottom
        :return: (dict) rescaled bbox
        """
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
        """
        Find iou based occlusion between two bboxes. That is,
        between given current bbox and and all other bboxes in the frame
        :param current_det: (dict) given detection
        :param all_detections: (list)  list of all other detections in the frame
        :return: 0 or 1, if occluded or not
        """
        # current bbox from given detection
        bbox = current_det['norm_bounding_box']
        current_crunch = [int(bbox['left']), int(bbox['top']), int(bbox['right']), int(bbox['bottom'])]

        # gather all the candidate bboxes from other detections in frame
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
            if iou > self.person_person_occ_thresh:  # if iou overlap is greater than threshold, then occ is true
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
            if self.frame_index % 10000 == 0: print(f"{self.frame_index} / {total_frame}")
            # get the detections of the frame
            detections = self.cpp_json["outputs"][self.frame_index]["detections"]
            # check if the detection has track id and pose
            detections = [det for det in detections if 'id' in det.keys() and 'body_skeleton' in det.keys()
                          and len(det['body_skeleton']) > 0]
            for ii in range(len(detections)):
                det = detections[ii]

                # get all data for the track for that frame
                frame_dict = self.fill_frame_dict(self.frame_index, det)
                # find if it is occluded by any other track, bbox iou based occlusion detection
                pp_occ = self.person_person_occlusion(det, detections)
                # combine person person occlusion with occlusion based on position of bbox in image
                frame_dict['occlusion'] = frame_dict['occlusion'] | pp_occ
                # update the frame dict with falling detection related information
                frame_dict = self.fill_frame_dict_fall_det_data(frame_dict.copy())

                # create track id in track dict for track analysis
                if det['id'] not in self.track_dict:  # then its the first frame of the track
                    self.track_dict[det['id']] = {}
                # update the frame data to the track dict
                self.track_dict[det['id']][self.frame_index] = frame_dict
                # only have the latest frames and remove the rest of history
                self.track_dict[det['id']] = {key: value for key, value in self.track_dict[det['id']].items()
                                              if key in range(self.frame_index - int(75),
                                                              self.frame_index + 1)}

                # detect falling
                falling_conf, bending_conf = self.falling_detector(self.track_dict[det['id']])
                if falling_conf > 0:
                    print(self.frame_index, det['id'], 'falling', np.round(falling_conf, 2))
                if bending_conf > 0:
                    print(self.frame_index, det['id'], 'bending', np.round(bending_conf, 2))
                # populate cpp json
                self.cpp_json["outputs"][self.frame_index]["detections"][ii]['falling_detection_conf'] = falling_conf
                self.cpp_json["outputs"][self.frame_index]["detections"][ii]['bending_detection_conf'] = bending_conf
                self.cpp_json["outputs"][self.frame_index]["detections"][ii]['occlusion'] = frame_dict['occlusion']
                duration = time.time() - start
                total_time = total_time + duration

        # save cpp json
        save_json(self.cpp_json_save_path, self.cpp_json)
        print('Total time ', total_time)

    def fill_frame_dict(self, frame_index, det):
        """
        Creates a dict having the frame data for a given detection of a frame. This frame data will have the
        information necessary for performing falling detection, such as center of gravity CoG and pose joints
        of detection with respect to different coordinates.
        :param frame_index: (int) current frame index
        :param det: (dict) current detection in the frame
        :return: (dict) frame dict
        """
        # initialise the frame dict
        frame_dict = {'trackid': det['id'], 'frame_index': frame_index}

        # extract bbox
        bbox = det['norm_bounding_box']
        frame_dict['bb_left'], frame_dict['bb_top'], frame_dict['bb_right'], frame_dict['bb_bottom'] = \
            int(bbox['left']), int(bbox['top']), int(bbox['right']), int(bbox['bottom'])

        # rescale the bbox to center the stickies and add padding. i.e., convert bbox to viddie box
        bbox_rescaled = self.rescale_bbox_for_stickies(bbox.copy())

        bbox_width_rescaled = bbox_rescaled['right'] - bbox_rescaled['left']
        bbox_height_rescaled = bbox_rescaled['bottom'] - bbox_rescaled['top']
        bbox_width = bbox['right'] - bbox['left']
        bbox_height = bbox['bottom'] - bbox['top']

        # extract pose joints or say key points
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

            # convert keypoint from frame to bbox rescaled
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

        # find CoG in different coord
        key_points_bbox = np.array(key_points_bbox)
        CoG_bbox = self.find_CoG_from_pose(key_points_bbox)  # CoG in bbox coord
        frame_dict['CoG_bbox_x'] = CoG_bbox[0]
        frame_dict['CoG_bbox_y'] = CoG_bbox[1]
        frame_dict['CoG_bbox_score'] = CoG_bbox[2]
        key_points_frame = np.array(key_points_frame)
        CoG_frame = self.find_CoG_from_pose(key_points_frame)  # CoG in frame coord
        frame_dict['CoG_frame_x'] = CoG_frame[0]
        frame_dict['CoG_frame_y'] = CoG_frame[1]
        frame_dict['CoG_frame_score'] = CoG_frame[2]
        key_points_norm = np.array(key_points_norm)
        CoG_norm = self.find_CoG_from_pose(key_points_norm)  # CoG in sticky size coord
        frame_dict['CoG_norm_x'] = CoG_norm[0]
        frame_dict['CoG_norm_y'] = CoG_norm[1]
        frame_dict['CoG_norm_score'] = CoG_norm[2]

        # manually define an occluded zone according to camera setup
        # if 10 < bbox['left'] and bbox['right'] < self.frame_width-10 and 200 < bbox['bottom'] < self.frame_height - 10:
        #     occlusion = 0
        # else:
        #     occlusion = 1
        occlusion = 0
        frame_dict['occlusion'] = occlusion

        return frame_dict

    def fill_frame_dict_fall_det_data(self, frame_dict):
        """
        Updates the frame dict with all specific information necessary for performing falling detection
        @param frame_dict: (dict) frame dict having the frame data for a given detection of a frame
        @return: (dict) updated frame dict
        """

        # occlusion condition based on position of bbox in image
        occlusion_cond = frame_dict['occlusion'] == 1

        # find shoulder and hip middle
        frame_dict['shoulder_middle_x'] = (frame_dict['shoulder_left_x'] + frame_dict['shoulder_right_x']) / 2
        frame_dict['shoulder_middle_y'] = (frame_dict['shoulder_left_y'] + frame_dict['shoulder_right_y']) / 2
        frame_dict['shoulder_middle_score'] = np.mean([frame_dict['shoulder_left_score'],
                                                       frame_dict['shoulder_right_score']], axis=0)

        frame_dict['hip_middle_x'] = (frame_dict['hip_left_x'] + frame_dict['hip_right_x']) / 2
        frame_dict['hip_middle_y'] = (frame_dict['hip_left_y'] + frame_dict['hip_right_y']) / 2
        frame_dict['hip_middle_score'] = np.mean([frame_dict['hip_left_score'], frame_dict['hip_right_score']], axis=0)

        # distance between shoulder middle and hip middle
        frame_dict['dist_shoulder_hip_y'] = frame_dict['shoulder_middle_y'] - frame_dict['hip_middle_y']
        frame_dict['dist_shoulder_hip_score'] = np.mean([frame_dict['shoulder_middle_score'],
                                                         frame_dict['hip_middle_score']], axis=0)

        # some filtering based on score and occlusion
        if (frame_dict['dist_shoulder_hip_score'] < self.kp_thresh) | (occlusion_cond):
            frame_dict['dist_shoulder_hip_y'] = np.nan

        # find trunk angle
        frame_dict['trunk_angle_y'] = np.rad2deg(
            np.arctan2(frame_dict['shoulder_middle_y'] - frame_dict['hip_middle_y'],
                       frame_dict['shoulder_middle_x'] - frame_dict['hip_middle_x']))
        frame_dict['trunk_angle_score'] = frame_dict['dist_shoulder_hip_score']

        # some filtering
        if (frame_dict['trunk_angle_score'] < self.kp_thresh) | (occlusion_cond):
            frame_dict['trunk_angle_y'] = np.nan

        # # # find occlusion of legs
        # we see whether the leg joint is poking outside the bbox in y direction, i.e. vertical direction.
        # If yes, then the legs cant be seen or say occluded
        # check whether left/right leg has low score
        ankle_left_score_low_condition = (frame_dict['ankle_left_score'] < self.kp_thresh)
        ankle_right_score_low_condition = (frame_dict['ankle_right_score'] < self.kp_thresh)

        # if both left and right ankle joints have high scores, take the maximum of left and right
        # if not choose the ankle with highest score. if none have high score, then it is sure occluded
        frame_dict['ankle_max_actual_bbox_y'] = np.nan
        if (not ankle_left_score_low_condition) and (not ankle_right_score_low_condition):
            frame_dict['ankle_max_actual_bbox_y'] = \
                np.maximum(frame_dict['ankle_left_actual_bbox_y'], frame_dict['ankle_right_actual_bbox_y'])
        if (not ankle_left_score_low_condition) and ankle_right_score_low_condition:
            frame_dict['ankle_max_actual_bbox_y'] = frame_dict['ankle_left_actual_bbox_y']
        if ankle_left_score_low_condition and (not ankle_right_score_low_condition):
            frame_dict['ankle_max_actual_bbox_y'] = frame_dict['ankle_right_actual_bbox_y']

        enlarged_bbox_height = self.sticky_size[1]  # enlarged bbox height
        # difference between the ankle and bbox bottom line in y direction
        frame_dict['ankle_from_actual_bbox_bottom_y'] = frame_dict['ankle_max_actual_bbox_y'] - enlarged_bbox_height
        frame_dict['ankle_from_actual_bbox_bottom_score'] = np.maximum(frame_dict['ankle_left_score'],
                                                                       frame_dict['ankle_right_score'])

        # legs based occlusion
        frame_dict['leg_occlusion'] = 0
        # if both ankles have low scores, then legs are occluded
        if ankle_left_score_low_condition & ankle_right_score_low_condition:
            frame_dict['leg_occlusion'] = 1
        # if the difference is greater than a threshold, then legs are occluded
        if frame_dict['ankle_from_actual_bbox_bottom_y'] > 15:
            frame_dict['leg_occlusion'] = 1

        # update occlusion condition with leg occlusion
        occlusion_cond = (frame_dict['occlusion'] == 1) | (frame_dict['leg_occlusion'] == 1)

        # CoG filtering
        if (frame_dict['CoG_norm_score'] < self.kp_thresh) | (occlusion_cond):
            frame_dict['CoG_norm_x'], frame_dict['CoG_norm_y'] = np.nan, np.nan

        # find CoG_bbox of legs
        frame_dict['hip_middle_bbox_x'] = (frame_dict['hip_left_bbox_x'] + frame_dict['hip_right_bbox_x']) / 2
        frame_dict['hip_middle_bbox_y'] = (frame_dict['hip_left_bbox_y'] + frame_dict['hip_right_bbox_y']) / 2
        frame_dict['hip_middle_bbox_score'] = np.mean([frame_dict['hip_left_bbox_score'],
                                                       frame_dict['hip_right_bbox_score']], axis=0)
        # we use the following joints to find CoG of legs in bbox coord
        dist_x, dist_y, score = [], [], []
        for joint in ['ankle_left', 'ankle_right', 'knee_left', 'knee_right', 'hip_left', 'hip_right']:
            frame_dict['dist_' + joint + '_x'] = frame_dict[joint + '_bbox_x'] - frame_dict['hip_middle_bbox_x']
            frame_dict['dist_' + joint + '_y'] = frame_dict[joint + '_bbox_y'] - frame_dict['hip_middle_bbox_y']
            frame_dict['dist_' + joint + '_score'] = np.minimum(frame_dict[joint + '_bbox_score'],
                                                                frame_dict['hip_middle_bbox_score'])

            dist_x.append(frame_dict['dist_' + joint + '_x'])
            dist_y.append(frame_dict['dist_' + joint + '_y'])
            score.append(frame_dict[joint + '_score'])

        frame_dict['CoG_bbox_legs_x'] = np.mean(dist_x, axis=0)
        frame_dict['CoG_bbox_legs_y'] = np.mean(dist_y, axis=0)
        frame_dict['CoG_bbox_legs_score'] = np.mean(score, axis=0)
        # filtering
        if (frame_dict['CoG_bbox_legs_score'] < self.kp_thresh) | (occlusion_cond):
            frame_dict['CoG_bbox_legs_x'], frame_dict['CoG_bbox_legs_y'] = np.nan, np.nan

        return frame_dict

    def variability(self, history_array):
        """
        Finds the variability/std of a given 1D array for a particular time window, ignoring Nans.
        @param history_array: (array) 1D array has the information of a series of frames of a track
        @return: (float) variability/std
        """
        # if the length of the array is greater than a minimum threshold and the array has at least certain number of
        # non Nan values in the chosen time window, then find variability, else return Nan
        if len(history_array) >= self.delta_min and sum(~np.isnan(history_array[-self.var_delta:])) >= self.delta_min:
            var = np.nanstd(history_array[-self.var_delta:], ddof=1)
        else:
            var = np.nan
        return var

    def smoothing(self, history_array):
        """
        Finds the mean of a given 1D array for a particular time window, ignoring Nans.
        @param history_array: (array) 1D array has the information of a series of frames of a track
        @return: (float) mean of the given array
        """
        # if the length of the array is greater than a minimum threshold and the array has at least certain number of
        # non Nan values in the chosen time window, then find mean, else return Nan
        if len(history_array) >= self.smooth_window and \
                sum(~np.isnan(history_array[-self.smooth_window:])) >= self.smooth_window:
            smoothed = np.nanmean(history_array[-self.smooth_window:])
        else:
            smoothed = np.nan
        return smoothed

    def create_list_from_dict(self, track_framedict, variable_name, frame_index_min, frame_index_max):
        """
        Given a variable_name of the track frame dict, create a 1D numpy array where each row
        represents a frame index and the value will be the variable data of that frame index.
        Fill in with nans for frame indexes that are missing data
        @param track_framedict: (dict) key is frame index, value is frame dict
        @param variable_name: (str) name of the variable to create list for from frame dict
        @param frame_index_min: (int) min of track_framedict keys
        @param frame_index_max: (int) max of track_framedict keys
        @return: (array) 1D numpy array
        """

        variable_name_list = np.array([track_framedict[frame_index][variable_name]
                                       if frame_index in track_framedict.keys() else np.nan
                                       for frame_index in range(frame_index_min, frame_index_max + 1)])
        return variable_name_list

    def falling_detector(self, track_framedict):
        """
        Given a history of pose joints, CoG and other related info over time for a track, this function detects falling
        and bending actions
        :param track_framedict: (dict) This is a dict, where keys are frames numbers
        and values are frame dicts for a given track. The keys range from current frame to a certain number of
        frames in the past. Basically a frame data history for the track
        :return: (float 0 to 1), (float 0 to 1) falling confidence and bending confidence
        """
        list_of_frame_index = list(track_framedict.keys())  # all the available frame indexes for the track
        frame_index_min = np.min(list_of_frame_index)
        frame_index_max = np.max(list_of_frame_index)

        # # # bending detection
        # some smoothing
        dist_shoulder_hip_y_list = self.create_list_from_dict(track_framedict, 'dist_shoulder_hip_y', frame_index_min,
                                                              frame_index_max)
        dist_shoulder_hip_y = self.smoothing(dist_shoulder_hip_y_list)

        trunk_angle_y_list = self.create_list_from_dict(track_framedict, 'trunk_angle_y', frame_index_min,
                                                        frame_index_max)
        trunk_angle_y = self.smoothing(trunk_angle_y_list)

        # find bending based on trunk angle and distance between shoulder and hip middle
        # (dist_shoulder_hip_y condition is redundant here. SOLVE THIS LATER)
        trunk_angle_condition = ((trunk_angle_y > -70) | (trunk_angle_y < -120))
        bending_condition = ((dist_shoulder_hip_y > -44) & trunk_angle_condition) | trunk_angle_condition
        track_framedict[frame_index_max]['bending'] = int(bending_condition)

        # dilate bending
        bending_list = self.create_list_from_dict(track_framedict, 'bending', frame_index_min, frame_index_max)
        if len(bending_list) >= self.delta_min and \
                sum(~np.isnan(bending_list[-self.bending_dilate_delta:])) >= self.delta_min:
            track_framedict[frame_index_max]['dilated_bending'] = self.dilate_bending(
                bending_list[-self.bending_dilate_delta:])
        else:
            track_framedict[frame_index_max]['dilated_bending'] = np.nan

        # # # trip fall detection
        # CoG smoothing
        CoG_norm_y_list = self.create_list_from_dict(track_framedict, 'CoG_norm_y', frame_index_min, frame_index_max)
        track_framedict[frame_index_max]['smooth_CoG_norm_y'] = self.smoothing(CoG_norm_y_list)

        smooth_CoG_norm_y_list = self.create_list_from_dict(track_framedict, 'smooth_CoG_norm_y', frame_index_min,
                                                            frame_index_max)
        # variability/std of CoG
        var_CoG_norm_y = self.variability(smooth_CoG_norm_y_list)

        # velocity/difference of CoG between current frame and a past frame
        if len(smooth_CoG_norm_y_list) >= 1 + self.vel_delta:
            vel_CoG_norm_y = smooth_CoG_norm_y_list[-1] - smooth_CoG_norm_y_list[-1 - self.vel_delta]
        else:
            vel_CoG_norm_y = np.nan

        # CoG_bbox of legs smoothing
        CoG_bbox_legs_y_list = self.create_list_from_dict(track_framedict, 'CoG_bbox_legs_y', frame_index_min,
                                                          frame_index_max)
        track_framedict[frame_index_max]['smooth_CoG_bbox_legs_y'] = self.smoothing(CoG_bbox_legs_y_list)

        # variability of CoG_bbox of legs
        smooth_CoG_bbox_legs_y_list = self.create_list_from_dict(track_framedict, 'smooth_CoG_bbox_legs_y',
                                                                 frame_index_min,
                                                                 frame_index_max)
        var_CoG_bbox_legs_y = self.variability(smooth_CoG_bbox_legs_y_list)

        # falling/tripping detection based on var and vel of CoG and var of CoG_bbox of legs
        detection_condition = (np.abs(var_CoG_norm_y) > 4) | \
                              (np.abs(vel_CoG_norm_y) > 8) | \
                              (np.abs(var_CoG_bbox_legs_y) > 4)
        track_framedict[frame_index_max]['detection'] = int(detection_condition)

        # condition on var_CoG_bbox_legs to remove falling/tripping false positives
        var_CoG_bbox_legs = (np.abs(var_CoG_bbox_legs_y) >= 2).astype(int)
        track_framedict[frame_index_max]['var_CoG_bbox_legs'] = var_CoG_bbox_legs
        # dilate var_CoG_bbox_legs
        var_CoG_bbox_legs_list = self.create_list_from_dict(track_framedict, 'var_CoG_bbox_legs', frame_index_min,
                                                            frame_index_max)
        if len(var_CoG_bbox_legs_list) >= self.delta_min and \
                sum(~np.isnan(var_CoG_bbox_legs_list[-self.CoG_dilate_delta:])) >= self.delta_min:
            var_CoG_bbox_legs = self.dilate_var_CoG(var_CoG_bbox_legs_list[-self.CoG_dilate_delta:])
        else:
            var_CoG_bbox_legs = np.nan

        # remove false positives from detection using dilated var_CoG_bbox_legs
        # i.e., if var_CoG_bbox_legs is 0, then it is not falling/tripping for sure
        if np.abs(var_CoG_bbox_legs) == 0:
            track_framedict[frame_index_max]['detection'] = 0

        # remove bending from falling detection
        if track_framedict[frame_index_max]['dilated_bending'] == 1:
            track_framedict[frame_index_max]['detection'] = 0

        # find the detection confidence
        # take a window of length conf_delta starting from current frame,
        # sum all falling detection frames in that window and normalise it with window size
        detection_list = self.create_list_from_dict(track_framedict, 'detection', frame_index_min, frame_index_max)

        # fill in nans/missing frames with interpolation of nearest frames
        mask = np.isnan(detection_list)
        if 0 < sum(mask) < len(detection_list):
            detection_list[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), detection_list[~mask])
            detection_list = np.round(detection_list)

        limit = np.min([len(detection_list), self.conf_delta])
        detection_conf = np.nansum(detection_list[-limit:]) / self.conf_delta
        if np.isnan(detection_conf): detection_conf = 0
        detection_conf = np.round(detection_conf, 2)

        # find the bending confidence same as before
        dilated_bending_list = self.create_list_from_dict(track_framedict, 'dilated_bending', frame_index_min,
                                                          frame_index_max)

        # fill in nans
        mask = np.isnan(dilated_bending_list)
        if 0 < sum(mask) < len(dilated_bending_list):
            dilated_bending_list[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask),
                                                   dilated_bending_list[~mask])
            dilated_bending_list = np.round(dilated_bending_list)

        limit = np.min([len(dilated_bending_list), self.conf_delta])
        bending_conf = np.nansum(dilated_bending_list[-limit:]) / self.conf_delta
        if np.isnan(bending_conf): bending_conf = 0
        bending_conf = np.round(bending_conf, 2)

        # one more check to remove bending false positives
        if bending_conf > 0:
            detection_conf = 0

        return detection_conf, bending_conf

    def find_CoG_from_pose(self, keypoints, all_joints=False):
        """
        Finds center of gravity CoG from pose joint positions
        :param keypoints: (list) list of pose joint positions' x, y pixel values
        :param all_joints: (bool) whether to include all joints in CoG calculation.
        If false, then head is not included in CoG calculation as it tends to  be occluded or noisy
        :return: (list) CoG's x, y, score
        """

        # whether to include include heads for CoG calculation
        if all_joints:
            start_index = 0
        else:
            start_index = 5

        # if there are no pose joints
        if len(keypoints) == 0:
            return [np.nan, np.nan, np.nan]

        # we define the CoG with respect to middle of the hip joints
        hip_middle = [int((keypoints[11, 0] + keypoints[12, 0]) / 2), int((keypoints[11, 1] + keypoints[12, 1]) / 2)]
        # find the distance between all joints from hip middle and take an average of the distances. That's the CoG
        dist_x, dist_y = [], []
        for point in keypoints[start_index:, :]:
            dist_x.append(point[0] - hip_middle[0])
            dist_y.append(point[1] - hip_middle[1])
        mean_score = np.mean(keypoints[start_index:, 2])  # mean score of all joints is taken as CoG's score
        CoG = [int(np.mean(dist_x)), int(np.mean(dist_y)), mean_score]

        return CoG
