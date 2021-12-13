# falling_down_detector
To detect trips/slips/fall/bending in a given video. 

## Description
Given a video, the video is first run through the cpp pipeline
to get the object detection, tracking and 2D pose. Distance is not used at the moment 
by the algorithm. Once we have the output from the cpp pipeline as a json file, 
we run this repository to detect trips/slips/fall/bending in that video. 
The detected trips/slips/fall/bending are updated in the cpp output json file and saved.

    
## Dependencies
Install the libraries in requirements.txt file 

Add the project to the PYTHONPATH to be able to run the scripts directly and to avoid any relative import errors

## Installing
First, clone the repo. Then take note of the following folder structure to run the falling detector.
Make sure the video folder, video name, cpp json names follow the below naming conventions.
If it doesn't, then the code will not recognise the files and will skip them.

#### Folder and name structure
- video_root--> path that stores all video folders
  - Video folder--> folder with video data (name same as video_name)
    - Video (video_name + '.mp4') 
    - Video's cpp json (video_name + '.json'). This is the output of the cpp pipeline
    - Falling detection output json--> (video_name + '_output.json'). 
    This is produced from the algorithm

## Executing falling detector
Run the code main.py. Set the root folder of all the videos in video_root variable.
For each video in the video_root, FallingDetector class is initialised and run. 

FallingDetector class loops through the cpp json frame by frame,
performs falling detection and populates/saves cpp json with results. 
If the video is not found or the cpp json is not found,
the video is skipped. 

### Note 

If a person falls flat on the ground or amost reaches ground, then it will be
detected as bending at the moment. Only trips/slips are detected as falling.
So to find the true positives, it is advisable to go through both 
falling and bending tracks. 

### To be solved in future

Following are some common false positives
- Squats 
- Orange safety cones confused with person legs, hence leg pose joints
become disproportionate
- Track id switching, causes sudden change in appearance
- Occlusion by objects, confuses pose sometimes

Following are some common false negatives
- Missing detections, people's grey dress is confused with background
or frame gets blurred/smudged patch by patch 

In the algorithm's bending detection, distance between shoulder and hip in y direction and trunk angle are the conditions
found. But distance between shoulder and hip condition is redundant in the current version (i.e. it is not really used. 
Only trunk condition is used.) Solve this later. 

## Authors
Yazhini Chitra Pradeep

