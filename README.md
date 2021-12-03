# falling_down_detector
To detect trips/slips/fall/bending in a given video. 

The video is first run through the cpp pipeline
to get the detection, tracking and pose. Distance is not used at the moment 
by the algorithm. 

#### Folder and name structure
- video_root--> path that stores all video folders
  - Video folder--> folder with video data (name same as video_name)
    - Video (video_name + '.mp4') 
    - Video's cpp json (video_name + '.json')
    - Falling detection output json--> (video_name + '_output.json'). 
    This is produced from the algorithm
## Requirements
Install the libraries in requirements.txt file   

## Run falling detector
Run the code main.py. Set the root folder of all the videos in video_root variable.
For each video in the video_root, frame_looper class is initialised and run. 

This frame_looper.py loops through the cpp json frame by frame,
performs falling detection and populates/saves cpp json with results. 
If the video is not found or the cpp json is not found,
the video is skipped. 

**Note:** 

If a person falls flat on the ground or amost reaches ground, then it will be
detected as bending at the moment. Only trips/slips are detected as falling.
So to find the true positives, it is advisable to go through both 
falling and bending tracks. 

**To be solved in future:** 

Following are some common false positives
- Squats 
- Orange safety cones confused with person legs, hence leg pose joints
become disproportionate
- Track id switching, causes sudden change in appearance
- Occlusion by objects, confuses pose sometimes

Following are some common false negatives
- Missing detections, people's grey dress is confused with background
or frame gets blurred/smudged patch by patch 

