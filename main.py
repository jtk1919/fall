import os
import time
from frame_looper import FrameLooper


"""
This is the main function. Change the video root. 
"""

video_root = '/home/yazhini/dB/shared_videos/Kajima/20_hr/20_hr_camera2' # folder containing all the video

t0 = time.time()
for video_name in os.listdir(video_root):
    t1 = time.time()
    print(video_name, ' #####################################')
    frame_looper = FrameLooper(video_name, video_root)
    if frame_looper.video_found and frame_looper.cpp_json_found:
        frame_looper.run()
        t2 = time.time()
        print(f"Time: {t2 - t1} | TOTAL: {t2 - t0}")
    else:
        print("skipping this video...")
