import numpy as np
import array
from calclib.mp_data_extract_util import get_mp_position_timeseries

def calc_speed_on_weighting(points,frame_speed,pos1,weighting1,pos2=0,weighting2=0,pos3=0,weighting3=0,pos4=0,weighting4=0):
    #print("pos1_speed")
    pos1_speed=0
    if (len(points) < 2):
        return 0
    
    points_calc_1 = get_mp_position_timeseries(points, pos1)
    pos1_speed = calculate_speed(points_calc_1[-2], points_calc_1[-1],frame_speed)
    #print("pos1 speed no weight : " + str(pos1_speed))
    pos1_speed = pos1_speed * weighting1
    #print("pos1_speed : " + str(pos1_speed))
    pos2_speed=0
    pos3_speed=0
    pos4_speed=0

    if pos2 != 0 :
        points_calc_2 = get_mp_position_timeseries(points, pos2)
        pos2_speed = calculate_speed(points_calc_2[-2], points_calc_2[-1],frame_speed)
        pos2_speed = pos2_speed * weighting2
        #print("pos2_speed : " + str(pos2_speed))
    if pos3 != 0 :
        points_calc_3 = get_mp_position_timeseries(points, pos3)
        pos3_speed = calculate_speed(points_calc_3[-2], points_calc_3[-1],frame_speed)
        pos3_speed = pos3_speed * weighting3        
        #print("pos3_speed : " + str(pos3_speed) + " " + str(weighting3))
    if pos4 != 0 :
        points_calc_4 = get_mp_position_timeseries(points, pos4)
        pos4_speed = calculate_speed(points_calc_4[-2], points_calc_4[-1],frame_speed)
        pos4_speed = pos4_speed * weighting4        
        #print("pos4_speed : " + str(pos4_speed))

    speed = pos1_speed + pos2_speed + pos3_speed + pos4_speed

    return speed



#input the landmark of mediapipe
def calculate_angle(a,b,c):
    #a = np.array([a.x, a.y]) # First
    #b = np.array([b.x, b.y]) # Mid
    #c = np.array([c.x, c.y]) # End
    
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    
    if angle >180.0:
        angle = 360-angle
        
    return angle

def all_angle(points_a, points_b, points_c):

    prev_pt = []
    whole = []
    index = 0  
    for pt in points_b:
        if (len(prev_pt) == 0):
            prev_pt = pt
        else:
            whole = np.append(whole, calculate_angle(points_a[index], pt, points_c[index]))

        index += 1

    return whole

def calculate_speed(startpos, endpos, time):
    #print(startpos)
    #print(endpos)
    displacement = np.sqrt((startpos[0] - endpos[0]) ** 2 + (startpos[1] - endpos[1]) ** 2 + (startpos[2] - endpos[2]) ** 2)

    #print("calculate_speed displacement : " + str(displacement))

    speed = displacement / time if time != 0 else float('inf')  # Avoid division by zero

    #print("calculate_speed : " + str(speed))

    return speed

def calculate_2d_speed(startpos, endpos, time):
    #print(startpos)
    #print(endpos)
    displacement = np.sqrt((startpos[0] - endpos[0]) ** 2 + (startpos[1] - endpos[1]) ** 2 )

    #print("calculate_speed displacement : " + str(displacement))

    speed = displacement / time if time != 0 else float('inf')  # Avoid division by zero

    #print("calculate_speed : " + str(speed))

    return speed

def calculate_acceleration(x, y, time):

    change_of_speed = y - x
    acceleration = change_of_speed / time if time != 0 else float('inf')  # Avoid division by zero
    return acceleration

def calculate_acceleration_from_joints(landmarks_pos_array, time):
    
    result_list = list()

    prev_position = None
    prev_velocity = None
    acceleration = None
    velocity = None

    for pos in landmarks_pos_array:
        if (pos is not None):
            current_position = pos

        if prev_position is not None:
            # Calculate velocity as change in position
            velocity = calculate_speed(prev_position, current_position, time)

            if prev_velocity is not None:
                # Calculate acceleration as change in velocity
                acceleration = (velocity - prev_velocity) / time

        # Update previous values
        prev_velocity = velocity

        # Update the previous knee position
        prev_position = current_position
        if (acceleration is not None):
            result_list.append(acceleration)

    return result_list

def all_speed(points, frame_speed):

    prev_pt = []
    whole = []
    for pt in points:
        if (prev_pt is None or len(prev_pt) == 0):
            prev_pt = pt
        else:
            if (pt is not None and prev_pt is not None) :
                whole = np.append(whole, calculate_speed(prev_pt, pt, frame_speed))
                
            prev_pt = pt

    return whole

def all_acceleration(speed_points, frame_speed):
    prev_pt = 0
    whole = []
    for pt in speed_points:
        if ((prev_pt) == 0):
            prev_pt = pt
        else:
            whole = np.append(whole, calculate_acceleration(prev_pt, pt, frame_speed))
            prev_pt = pt

    return whole

def avg_speed(points, frame_speed):

    return np.average(all_speed(points, frame_speed))

def avg_speed_in_1_std(points, frame_speed):

    speeds = all_speed(points, frame_speed)

    # Calculate the standard deviation
    standard_deviation = np.std(speeds)
    mean = np.mean(speeds)

    #print(f'mean={mean} and sd={standard_deviation}')
    

    point_in_1_sd = []
    for pt in speeds:
        #print(pt)

        if pt <= mean + standard_deviation and pt >= mean - standard_deviation:
            point_in_1_sd = np.append(point_in_1_sd, pt) 


    return np.average(point_in_1_sd)

# input the world_landmark
def get_distinance_of_two_joint(points_a, points_b):

    p1 = points_a
    p2 = points_b

    squared_dist = np.sum((p1-p2)**2, axis=0)
    dist = np.sqrt(squared_dist)

    return dist


# input the world_landmark
def get_avg_distinance_of_two_joint(points_a, points_b):
    avg_array = []
    for index, item in enumerate(points_a):
        if (item is not None):
            dist = get_distinance_of_two_joint(item, points_b[index])
            avg_array = np.append(avg_array, dist) 
  
    return np.average(avg_array)


def get_joint_name(joint_number) :    
    if (joint_number == 1): return 'Nose'
    if (joint_number == 2): return 'Left Eye Inner'
    if (joint_number == 3): return 'Left Eye'
    if (joint_number == 4): return 'Left Eye Outer'
    if (joint_number == 5): return 'Right Eye Inner'
    if (joint_number == 6): return 'Right Eye'
    if (joint_number == 7): return 'Right Eye Outer'
    if (joint_number == 8): return 'Left Ear Tip'
    if (joint_number == 9): return 'Right Ear Tip'
    if (joint_number == 10): return 'Mouth Left'
    if (joint_number == 11): return 'Mouth Right'
    if (joint_number == 12): return 'Left Shoulder'
    if (joint_number == 13): return 'Right Shoulder'
    if (joint_number == 14): return 'Left Elbow'
    if (joint_number == 15): return 'Right Elbow'
    if (joint_number == 16): return 'Left Wrist'
    if (joint_number == 17): return 'Right Wrist'
    if (joint_number == 18): return 'Left Pinky'
    if (joint_number == 19): return 'Right Pinky'
    if (joint_number == 20): return 'Left Index'
    if (joint_number == 21): return 'Right Index'
    if (joint_number == 22): return 'Left Thumb'
    if (joint_number == 23): return 'Right Thumb'
    if (joint_number == 24): return 'Left Hip'
    if (joint_number == 25): return 'Right Hip'
    if (joint_number == 26): return 'Left Knee'
    if (joint_number == 27): return 'Right Knee'
    if (joint_number == 28): return 'Left Ankle'
    if (joint_number == 29): return 'Right Ankle'
    if (joint_number == 30): return 'Left Heel'
    if (joint_number == 31): return 'Right Heel'
    if (joint_number == 32): return 'Left Foot Index'
    if (joint_number == 33): return 'Right Foot Index'

