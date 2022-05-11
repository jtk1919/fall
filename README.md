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

Add the project path to the PYTHONPATH to be able to run the scripts directly

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



