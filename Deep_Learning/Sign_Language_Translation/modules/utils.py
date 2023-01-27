## mediapipe에서 제공하는 정규화 과정 -> 좌표값과 각도값을 넣어줌.

import os
import cv2
import numpy as np

def createDirectory(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print("Error: Failed to create the directory.")

# vector normalization
def Vector_Normalization(joint): # 각 사람들의 손 크기에 따라 달라지는 keypoint 값들을 보정해주기 위하여
    # Compute angles between joints
    v1 = joint[[0,1,2,3,0,5,6,7,0,9,10,11,0,13,14,15,0,17,18,19], :2] # Parent joint
    v2 = joint[[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20], :2] # Child joint
    v = v2 - v1 
    # Normalize v
    v = v / np.linalg.norm(v, axis=1)[:, np.newaxis]

    # Get angle using arcos of dot product
    angle = np.arccos(np.einsum('nt,nt->n',
        v[[0,1,2,4,5,6,8,9,10,12,13,14,16,17,18],:], 
        v[[1,2,3,5,6,7,9,10,11,13,14,15,17,18,19],:])) 

    angle = np.degrees(angle) # Convert radian to degree

    angle_label = np.array([angle], dtype=np.float32)

    return v, angle_label

# coordinate normalization(min-max scale)
def Coordinate_Normalization(joint):
    x_coordinates = []
    y_coordinates = []
    for i in range(21):
        x_coordinates.append(joint[i][0] - joint[0][0])
        y_coordinates.append(joint[i][1] - joint[0][1])
    for i in range(21):
        x_coordinates.append(joint[i+21][0] - joint[21][0])
        y_coordinates.append(joint[i+21][1] - joint[21][1])

    x_left_hand = x_coordinates[:21]
    x_right_hand = x_coordinates[21:]
    y_left_hand = y_coordinates[:21]
    y_right_hand = y_coordinates[21:]

    if max(x_left_hand) == min(x_left_hand):
        x_left_hand_scale = x_left_hand
    else:
        x_left_hand_scale = x_left_hand/(max(x_left_hand)-min(x_left_hand))
    
    if max(x_right_hand) == min(x_right_hand):
        x_right_hand_scale = x_right_hand
    else:
        x_right_hand_scale = x_right_hand/(max(x_right_hand)-min(x_right_hand))
    
    if max(y_left_hand) == min(y_left_hand):
        y_left_hand_scale = y_left_hand
    else:
        y_left_hand_scale = y_left_hand/(max(y_left_hand)-min(y_left_hand))
    
    if max(y_right_hand) == min(y_right_hand):
        y_right_hand_scale = y_right_hand
    else:
        y_right_hand_scale = y_right_hand/(max(y_right_hand)-min(y_right_hand))
            
    full_scale = np.concatenate([x_left_hand_scale.flatten(),
                                    x_right_hand_scale.flatten(),
                                    y_left_hand_scale.flatten(),
                                    y_right_hand_scale.flatten()])
    return full_scale
