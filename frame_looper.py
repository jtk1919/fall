import os
import cv2
import pandas as pd
import numpy as np

from utils import load_json, imshow_frame


class FrameLooper:

    def __init__(self, video_name, video_root):
        """
        Loops through frame
        @param video_name: (string) name of the video
        @param video_root: (string) folder in which the video is present
        """

        self.video_root = video_root
        self.video_name = video_name
        self.video_path = os.path.join(video_root, self.video_name.split('.')[0], self.video_name)

        # read cpp json output
        self.cpp_json_path = os.path.join(self.video_root, self.video_name.split('.')[0],
                                          self.video_name + '.json')
        self.cpp_json = None
        try:
            self.cpp_json = load_json(self.cpp_json_path)
        except Exception as e:
            print('could not find the cpp json output in ', self.cpp_json_path)
            print(e)
            pass

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


    def run(self):
        """
        Loops through the video frame by frame
        """

        while True:
            key = cv2.waitKey(1) & 0xFF  # Wait a millisecond to store any keyboard inputs
            if key == ord('p'):
                self.pause = not self.pause
            if key == ord('q') or key == 27 or key == ord('s'):
                break
            if not self.pause or key == ord(' '):
                # Read the next frame from the video capture depending on the data rate
                for _ in range(self.data_rate):
                    ret, frame = self.video_reader.read()
                    # Break so we don't skip the first frame.
                    if self.frame_index == 0:
                        break

                # if we have a frame
                if ret:
                    detections = self.cpp_json["outputs"][self.frame_index]["detections"]
                    self.raw_frame_index = self.cpp_json["outputs"][self.frame_index]['raw_index']

                    # do stuff here
                    print(self.frame_index, len(detections))
                    for track in detections:
                        # bbox_x1, bbox_y1, bbox_x2, bbox_y2 = [int(track['norm_bounding_box']['left']),
                        #                                       int(track['norm_bounding_box']['top']),
                        #                                       int(track['norm_bounding_box']['right']),
                        #                                       int(track['norm_bounding_box']['bottom'])
                        #                                       ]
                        # cv2.rectangle(frame, (int(bbox_x1), int(bbox_y1)), (int(bbox_x2), int(bbox_y2)),
                        #               (0,0,255), 3)
                        # cv2.putText(frame, str(track['id']), (int(bbox_x1), int(bbox_y1-10)), 2,
                        #             5e-3 * 170, (0, 255, 0), 2, lineType=cv2.LINE_AA)
                        pose = track['body_skeleton']
                    imshow_frame('frame', frame, int(self.frame_width), int(self.frame_height))


                    self.frame_index += 1

                else: # if the video returns no frame, then break
                    break



if __name__ == "__main__":

    video_root = '/home/yazhini/dB/shared_videos/Kajima'
    video_name = '2021-07-28_10-00-00.mkv'

    frame_looper = FrameLooper(video_name, video_root)
    frame_looper.run()