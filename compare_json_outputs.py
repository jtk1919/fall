from .utils import *

####
# compares the output of two jsons generated from the falling detection algorithm.
# Useful when we have json output from cloud and from local computer to see if they produce same thing.
# Also useful when merging/updating/refactoring algorithm code
####

# change the file paths below
file1_json = load_json('/home/yazhini/dB/shared_videos/Kajima/20_hr/test/'
                          '2021-10-15_16-00-00_no5_162637_slip/2021-10-15_16-00-00_no5_162637_slip_output.json')
file2_json = load_json('/home/yazhini/dB/shared_videos/Kajima/20_hr/test/2021-10-15_16-00-00_no5_162637_slip/'
                         '2021-10-15_16-00-00_no5_162637_slip_output_yazhini_branch.json')

total_frame = len(file1_json['outputs'])
falling_conf_mistakes = 0
bending_conf_mistakes = 0
for frame_index in range(len(file1_json["outputs"])):
    if frame_index % 10000 == 0: print(f"{frame_index} / {total_frame}")
    file1_detections = file1_json["outputs"][frame_index]["detections"]
    file2_detections = file2_json["outputs"][frame_index]["detections"]
    # if file1_detections != file2_detections:
    #     print(frame_index, 'number of detections in this frame messed up')
    for ii in range(len(file1_detections)):
        file1_det = file1_detections[ii]
        file2_det = file2_detections[ii]
        # check if conf values are same for falling and bending
        if 'falling_detection_conf' in file1_det.keys() and 'falling_detection_conf' in file2_det.keys():
            if file2_det['falling_detection_conf'] != file1_det['falling_detection_conf']:
                falling_conf_mistakes = falling_conf_mistakes + 1
                print(frame_index, file1_det['id'], file2_det['id'], 'falling det conf not agreeing')
                print(file1_det['falling_detection_conf'], file2_det['falling_detection_conf'])
            if file2_det['bending_detection_conf'] != file1_det['bending_detection_conf']:
                bending_conf_mistakes = bending_conf_mistakes + 1
                print(frame_index, file1_det['id'], file2_det['id'], 'bending det conf not agreeing')
                print(file1_det['bending_detection_conf'], file2_det['bending_detection_conf'])
        else:  # check if there is a falling output in both files
            if 'falling_detection_conf' in file1_det.keys() and 'falling_detection_conf' not in file2_det.keys():
                print(frame_index, file1_det['id'], file2_det['id'], 'falling det is missing in file2 json')
            if 'falling_detection_conf' in file2_det.keys() and 'falling_detection_conf' not in file1_det.keys():
                print(frame_index, file1_det['id'], file2_det['id'], 'falling det is missing in file1 json')

print('No of falling conf mistakes', falling_conf_mistakes)
print('No of bending conf mistakes', bending_conf_mistakes)