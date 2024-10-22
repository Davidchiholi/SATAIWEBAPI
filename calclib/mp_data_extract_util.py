import mediapipe as mp
import numpy as np
import pose_module as pm
import mediapipe.python.solutions.pose as mp_pose
import numpy.typing as npt

#mp_landmarks is the raw data from mediapipe
#mp_joint_name is the index of joint
def get_mp_position_timeseries(mp_landmarks, mp_joint_name):
    
    result_list = list()
  
    #print("get_mp_position_timeseries 1")
    #print(mp_landmarks)
    for pos in mp_landmarks:
        if pos is not None:
            result_list.append(pos[mp_joint_name])
        else:
            result_list.append(None)
      
    #print("get_mp_position_timeseries 2")
    return result_list

def read_exact_landmark_positions_2d(
    results: any,
    image_width: int,
    image_height: int,
) -> npt.NDArray[np.float32] | None:
    if results.pose_landmarks is None :
        return None
    else:
        pose_landmarks = results.pose_landmarks.landmark
        normalized_landmarks = [pose_landmarks[lm] for lm in mp_pose.PoseLandmark]
        return np.array([(image_width * lm.x, image_height * lm.y) for lm in normalized_landmarks])


def read_landmark_positions_3d(
    results: any,
) -> npt.NDArray[np.float32] | None:
    if results.pose_landmarks is None:
        return None
    else:
        pose_landmarks = results.pose_landmarks.landmark
        landmarks = [pose_landmarks[lm] for lm in mp_pose.PoseLandmark]
        return np.array([(lm.x, lm.y, lm.z) for lm in landmarks])


def read_world_landmark_positions_3d(
    results: any,
) -> npt.NDArray[np.float32] | None:
    if results.pose_world_landmarks is None:
        print("read_world_landmark_positions_3d is None")
        return None
    else:
        pose_landmarks = results.pose_world_landmarks.landmark
        landmarks = [pose_landmarks[lm] for lm in mp_pose.PoseLandmark]
        return np.array([(lm.x, lm.y, lm.z) for lm in landmarks])
