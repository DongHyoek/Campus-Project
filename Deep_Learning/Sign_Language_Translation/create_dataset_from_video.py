import cv2
import sys, os
import mediapipe as mp
import numpy as np
import modules.holistic_module as hm
from modules.utils import createDirectory
import json
import time


from modules.utils import Coordinate_Normalization, Vector_Normalization

createDirectory('dataset/output_video')

# 저장할 파일 이름
save_file_name = "train"

# 시퀀스의 길이(30 -> 10)
seq_length = 10


actions = ['ㄱ', 'ㄴ', 'ㄷ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅅ', 'ㅇ', 'ㅈ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ',
             'ㅏ', 'ㅑ', 'ㅓ', 'ㅕ', 'ㅗ', 'ㅛ', 'ㅜ', 'ㅠ', 'ㅡ', 'ㅣ',
             'ㅐ', 'ㅒ', 'ㅔ', 'ㅖ', 'ㅢ', 'ㅚ', 'ㅟ']

dataset = dict()

# 데이터 저장을 위한 dataset dictionary 생성. 각 라벨값마다 데이터를 저장함.
for i in range(len(actions)):
    dataset[i] = [] # 해당하는 key가 결국 label 값을 의미.

# 비디오를 하나씩 꺼내와서 Capture 
# MediaPipe holistic model -> 손의 랜드마크(21개)를 추적
detector = hm.HolisticDetector(min_detection_confidence=0.3)

videoFolderPath = "dataset/output_video"
videoTestList = os.listdir(videoFolderPath)

testTargetList =[]

created_time = int(time.time())

for videoPath in videoTestList:
    actionVideoPath = f'{videoFolderPath}/{videoPath}'
    actionVideoList = os.listdir(actionVideoPath)
    for actionVideo in actionVideoList:
        fullVideoPath = f'{actionVideoPath}/{actionVideo}'
        testTargetList.append(fullVideoPath)

print("---------- Start Video List ----------")
testTargetList = sorted(testTargetList, key=lambda x:x[x.find("/", 9)+1], reverse=True)
print(testTargetList)
print("----------  End Video List  ----------\n")

# 각 비디오마다 진행
for target in testTargetList:

    data = []
    first_index = target.find("/")
    second_index = target.find("/", first_index+1)
    third_index = target.find("/", second_index+1)
    idx = actions.index(target[target.find("/", second_index)+1:target.find("/", third_index)]) # 각 label의 인덱스를 저장.

    print("Now Streaming :", target)
    cap = cv2.VideoCapture(target) # video capture가 열림. -> 영상을 읽어오는 cv2의 함수 들어감.

    # 열렸는지 확인
    if not cap.isOpened():
        print("Camera open failed!")
        sys.exit()

    # 웹캠의 속성 값을 받아오기
    # 정수 형태로 변환하기 위해 round
    w = round(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f'w는{w}, h는 {h}입니다.') # 640, 480임.
    fps = cap.get(cv2.CAP_PROP_FPS) # 카메라에 따라 값이 정상적, 비정상적
    print('현재 fps:',fps)

    if fps != 0:
        delay = round(1000/fps) # 더 좋은 웹캠이 있으면 해당 프레임마다 받을 수 있도록 지정
    else:
        delay = round(1000/30) # default는 30프레임으로 지정

    # 프레임을 받아와서 저장하기
    while True: # 
        ret, img = cap.read() # ret, img로 영상이 쪼개어질 수 있음. ret에는 영상을 제대로 읽어왔는지 아닌지가 들어가고, image에 현재프레임.

        if not ret:
            break

        img = detector.findHolistic(img, draw=True) # 현재 프레임의 array에서 mediapipe의 detector를 이용해서 찾음.
        # _, left_hand_lmList = detector.findLefthandLandmark(img)
        _, right_hand_lmList = detector.findRighthandLandmark(img)

        
        # if len(left_hand_lmList) != 0 and len(right_hand_lmList) != 0:
        # if left_hand_lmList is not None or right_hand_lmList is not None:
        if right_hand_lmList is not None:
            joint = np.zeros((42, 2)) # 오른쪽, 왼손 joint
            
            # 왼손 랜드마크 리스트
            # for j, lm in enumerate(left_hand_lmList.landmark):
                # joint[j] = [lm.x, lm.y]
            
            # 오른손 랜드마크 리스트
            for j, lm in enumerate(right_hand_lmList.landmark):
                # joint[j+21] = [lm.x, lm.y]
                joint[j] = [lm.x, lm.y]

            # 좌표 정규화
            # full_scale = Coordinate_Normalization(joint)

            # 벡터 정규화
            vector, angle_label = Vector_Normalization(joint)

            # 정답 라벨링
            angle_label = np.append(angle_label, idx) # 여기에 정답값이 추가됨

            print(vector)
            print(angle_label)

            # 위치 종속성을 가지는 데이터 저장
            # d = np.concatenate([joint.flatten(), angle_label])

            # 벡터 정규화를 활용한 위치 종속성 제거
            d = np.concatenate([vector.flatten(), angle_label.flatten()])

            # 정규화 좌표를 활용한 위치 종속성 제거 
            #d = np.concatenate([full_scale, angle_label.flatten()])
            
            data.append(d) # d의 shape는 (56,) , data는 list
            
            
            

        

        # draw box
        # cv2.rectangle(img, (0,0), (w, 30), (245, 117, 16), -1)

        # draw text target name
        # cv2.putText(img, target, (15,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)


        # cv2.imshow('img', img)
        cv2.waitKey(delay)

        # esc를 누르면 강제 종료
        if cv2.waitKey(delay) == 27: 
            break

    print("\n---------- Finish Video Streaming ----------")
    print(len(data),'<--- 이전 data.shape')
    data = np.array(data)
    print(data.shape,'<--- 이후 data.shape')
    # Create sequence data
    ## 1~10 -> 2~11 -> 3~12 -> 4~13 -> ...
    # 10개의 sequence길이로 영상을 짜른 데이터값들이 총 len-seq_length 만큼 나온다. 그 데이터는 각각 keypoint와 angle point로 구성되어져있다.
    for seq in range(len(data) - seq_length): 
        dataset[idx].append(data[seq:seq + seq_length])    
    print(len(data) - seq_length)

for i in range(len(actions)):
    save_data = np.array(dataset[i])
    np.save(os.path.join('dataset', f'seq_{actions[i]}_{created_time}'), save_data)


print("\n---------- Finish Save Dataset ----------")

## 최종적으로는 손의 keypoint 위치값 + angle에 대한 값들이 seq_length 별로 들어가있음.
## 

