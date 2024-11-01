import array
import json
import cv2, time
import os
import random
import pose_module as pm
##new code begin
import numpy as np
import mediapipe as mp
##new code end
from calclib.mp_data_extract_util import read_exact_landmark_positions_2d
from calclib.mp_data_extract_util import get_mp_position_timeseries
from calclib.mp_data_extract_util import read_world_landmark_positions_3d
from calclib.motion_calc_util import avg_speed_in_1_std
from calclib.motion_calc_util import calculate_acceleration
from calclib.motion_calc_util import get_avg_distinance_of_two_joint
from calclib.motion_calc_util import calc_speed_on_weighting
from calclib.motion_calc_util import calculate_angle
from calclib.motion_calc_util import calculate_acceleration_from_joints
from calclib.motion_calc_util import get_joint_name
from scipy.spatial.distance import cosine
from fastdtw import fastdtw
from azure.storage.blob import ContainerClient
##new code begin
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
##new code end

def visualize(
        image,
        detection_result,
        catg_name_lookup
        ) -> np.ndarray:
    ##Draws bounding boxes on the input image and return it.
    ##   Args:
    ##        image: The input RGB image.
    ##        detection_result: The list of all "Detection" entities to be visualize.
    ##    Returns:
    ##        Image with bounding boxes.

        probability = -1
        detection_coords = ((0, 0),(0, 0))
		
        for detection in detection_result.detections:
        # Draw bounding_box
            # Draw label and score
            category = detection.categories[0]
            category_name = category.category_name
            if (catg_name_lookup.lower() == category_name.lower()):
                probability = round(category.score, 2)
                if (probability > 0.20):
                    bbox = detection.bounding_box
                    start_point = bbox.origin_x, bbox.origin_y
                    end_point = bbox.origin_x + bbox.width, bbox.origin_y + bbox.height
                    detection_coords = ((bbox.origin_x, bbox.origin_y),
												(bbox.origin_x + bbox.width, bbox.origin_y + bbox.height))
                    cv2.rectangle(image, start_point, end_point, cv2.COLOR_BGR2RGB, 3)
                #    result_text = category_name + ' (' + str(probability) + ')'
                #    text_location = (MARGIN + bbox.origin_x,
                #                 MARGIN + ROW_SIZE + bbox.origin_y)
                #    cv2.putText(image, result_text, text_location, cv2.FONT_HERSHEY_PLAIN,
                #        FONT_SIZE, TEXT_COLOR, FONT_THICKNESS)
                else:
                    probability=-1
                    detection_coords = ((0, 0),(0, 0))
                break

        return image , detection_coords, probability

def compare_positions(benchmark_video, user_video, benchmark_blobcontainer, user_blobcontainer, output_filename, output_fullname, blob_containername, check_rate, blob_connection, sport, show_window, combine_result, deleted_blob, model, equip, model1, equip1, model2, equip2, joints_dict):
	if (benchmark_video=='' or user_video==''):
		return 0

	newBenchmark_video = ''
	newUser_video = ''
	if (benchmark_blobcontainer != ''):
		benchmarkcontainer_client = ContainerClient.from_connection_string(conn_str=blob_connection, container_name=benchmark_blobcontainer)
		newBenchmark_video = 'V'+str(time.strftime("%Y%m%d%H%M%S"))+str(random.randint(0,199999999)) + os.path.basename(benchmark_video)  
		with open(file=newBenchmark_video, mode="wb") as download_filebm:
			download_filebm.write(benchmarkcontainer_client.download_blob(benchmark_video).readall())
	
	if (user_blobcontainer != ''):
		usercontainer_client = ContainerClient.from_connection_string(conn_str=blob_connection, container_name=user_blobcontainer)
		newUser_video = 'V'+str(time.strftime("%Y%m%d%H%M%S"))+str(random.randint(0,199999999)) + os.path.basename(user_video)  
		with open(file=newUser_video, mode="wb") as download_fileur:
			download_fileur.write(usercontainer_client.download_blob(user_video).readall())

	if (newBenchmark_video == ''):
		newBenchmark_video = benchmark_video

	if (newUser_video == ''):
		newUser_video = user_video

	#print("video input:" + newBenchmark_video + " " + newUser_video)

	benchmark_cam = cv2.VideoCapture(newBenchmark_video)
	user_cam = cv2.VideoCapture(newUser_video) 
	w2 = int(benchmark_cam.get(cv2.CAP_PROP_FRAME_WIDTH))
	h2 = int(benchmark_cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
	w1 = int(user_cam.get(cv2.CAP_PROP_FRAME_WIDTH))
	h1 = int(user_cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
	#ADD
	mp_pose = mp.solutions.pose 
	pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) 
	all_2d_points_user = list()
	all_3d_world_points_user = list()
	all_2d_points_benchmark = list()
	all_3d_world_points_benchmark = list()
	correct_frames_ratio_list = list()
	prev_speed_user = 0
	frame_fps_user = int(user_cam.get(5))  #fps
	frame_fps_benchmark = int(benchmark_cam.get(5))  #fps
	#print("frame_fps:")
	#print(frame_fps_user)
	#print(frame_fps_benchmark)
	#ADD===END

	if h1 > w1:
		nh1 = 720
		nw1 = int(w1/h1*nh1)
		nh2 = 720
		nw2 = int(w2/h2*nh2)
	else:
		nw1 = 720
		nh1 = int(h1/w1*nw1)
		nw2 = 720
		nh2 = int(h2/w2*nw2)
	fourcc = cv2.VideoWriter_fourcc(*'mp4v')
	if (output_filename!=''):
		if (combine_result):
			if h1 > w1:
				out = cv2.VideoWriter(output_filename, fourcc, 25, (nw1+nw2, nh1))
			else:
				out = cv2.VideoWriter(output_filename, fourcc, 25, (nw1, nh1+nh2))
		else:
			out = cv2.VideoWriter(output_filename, fourcc, 25, (nw1, nh1))
			
	fps_time = 0 #Initializing fps to 0

##new code
	if model != '':
		base_options = python.BaseOptions(model_asset_path=model)
		options = vision.ObjectDetectorOptions(base_options=base_options,
									score_threshold=0.5)
		mp_object_detection = vision.ObjectDetector.create_from_options(options)
#	else:
#		base_options = python.BaseOptions(model_asset_path='efficientdet.tflite')
#	options = vision.ObjectDetectorOptions(base_options=base_options,
#									score_threshold=0.5)
#	mp_object_detection = vision.ObjectDetector.create_from_options(options)
##new code

	detector_1 = pm.poseDetector()
	detector_2 = pm.poseDetector()
	detector_2F = pm.poseDetector()
	benchmark_count=0
	if (benchmark_cam.isOpened()):
		benchmark_count = benchmark_cam.get(cv2.CAP_PROP_FRAME_COUNT)
	if (user_cam.isOpened()):
		user_count = user_cam.get(cv2.CAP_PROP_FRAME_COUNT)
	frame_counter = 0
	correct_frames = 0
	current_frames_count = 0
	saved_frames =0
	if check_rate > 0.5 or check_rate < 0.01:
		check_rate = 0.1

	while (benchmark_cam.isOpened() or user_cam.isOpened()):
		try:
			ret_val, image_1 = user_cam.read()
			#Loop the video if it ended. If the last frame is reached, reset the capture and the frame_counter
			if frame_counter == user_cam.get(cv2.CAP_PROP_FRAME_COUNT):
				break
		

#				frame_counter = 0 #Or whatever as long as it is the same as next line
#				correct_frames = 0
#				user_cam.set(cv2.CAP_PROP_POS_FRAMES, 0)

#			if show_window:
#				winname = "User Video"
#				cv2.namedWindow(winname)		   # Create a named window
#				cv2.moveWindow(winname, 720,-100)  # Move it to desired location

			image_user_org = image_1
			image_1 = cv2.resize(image_1, (nw1,nh1))
			image_1 = detector_1.findPose(image_1)
			lmList_user = detector_1.findPosition(image_1)
			del lmList_user[1:11]

			#Loop the video if it ended. If the last frame is reached, reset the capture and the frame_counter
			if frame_counter == benchmark_cam.get(cv2.CAP_PROP_FRAME_COUNT):
				break
#				frame_counter = 0 #Or whatever as long as it is the same as next line
#				correct_frames = 0
#				benchmark_cam.set(cv2.CAP_PROP_POS_FRAMES, 0)

			ret_val_1,image_2 = benchmark_cam.read()
			image_benchmark_org = image_2
			image_2 = cv2.resize(image_2, (nw2,nh2))
			image_2F = cv2.flip(image_2, 1)
			image_2 = detector_2.findPose(image_2)
			image_2F = detector_2F.findPose(image_2F)
			lmList_benchmark = detector_2.findPosition(image_2)
			lmList_benchmark_F = detector_2F.findPosition(image_2F)

			is_enable_face_lankmark = joints_dict.get("enable_face_landmark")
			if (is_enable_face_lankmark is not None and is_enable_face_lankmark > 0):
				# not delete the face landmarks
				pass
			else:
				del lmList_benchmark[1:11]
				del lmList_benchmark_F[1:11]

			frame_counter += 1
			print("===================start landamrk: frame=" + str(current_frames_count))
			#ADD===
			results = pose.process(image_user_org)
			#print("===================start landamrk 1.1:")
			#print(results)
			allpoints = (read_exact_landmark_positions_2d(results, w1, h1))
			#print("===================start landamrk user 2D:")
			#print(allpoints)
			allpoints_3d_world = (read_world_landmark_positions_3d(results))
			#print("===================start landamrk user 3D:")
			#print(allpoints_3d_world)
			all_2d_points_user.append(allpoints)
			all_3d_world_points_user.append(allpoints_3d_world)
			#print("===================start landamrk3:")
			results = pose.process(image_benchmark_org)
			allpoints = (read_exact_landmark_positions_2d(results, w2, h2))
			#print("===================start landamrk4:")
			allpoints_3d_world = (read_world_landmark_positions_3d(results))
			#print("===================start landamrk5:")
			all_2d_points_benchmark.append(allpoints)
			all_3d_world_points_benchmark.append(allpoints_3d_world)
			#print("===================start landamrk6:")
			#input joints point & weighting to calc speed
			try :
				speed_user = calc_speed_on_weighting(all_3d_world_points_user, 1/frame_fps_user, joints_dict["joint1"], joints_dict["joint1_weighting"],joints_dict["joint2"], joints_dict["joint2_weighting"], joints_dict["joint3"], joints_dict["joint3_weighting"], joints_dict["joint4"], joints_dict["joint4_weighting"])
			except Exception as error:
				print("An exception at calc:", error) # An exception occurred: division by zero
				pass
			#print("===================speed:")
			
			#print(speed_user)
			#print("===================acceleration:")
			acceleration_user = calculate_acceleration(speed_user , prev_speed_user, 1/frame_fps_user)
			#print(acceleration_user)			
			prev_speed_user = speed_user
			#ADD===END


			if ret_val_1 or ret_val:
				error, _ = fastdtw(lmList_user, lmList_benchmark, dist=cosine)
				error_F, _ = fastdtw(lmList_user, lmList_benchmark_F, dist=cosine)
				error_Show = error
				if error_F < error:
					error_Show = error_F

				#newcoding begin
				if model != '':
					# Initialize the MediaPipe Object Detector with the SSD MobileNetV2 model

					rgb_image1 = cv2.cvtColor(image_1, cv2.COLOR_BGR2RGB)
					mp_image1 = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image1)
					resultsobj1 = mp_object_detection.detect(mp_image1)
					
					rgb_image2 = cv2.cvtColor(image_2, cv2.COLOR_BGR2RGB)
					mp_image2 = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image2)
					resultsobj2 = mp_object_detection.detect(mp_image2)
					
					image_1, detection1_coords , probability1 = visualize(image_1,resultsobj1,equip)
					image_2, detection2_coords , probability2 = visualize(image_2,resultsobj2,equip)
								
					if probability1 != -1 and probability2 != -1:
						# Compute the distance between the two detections using FastDTW
						error_equip, _ = fastdtw(detection1_coords, detection2_coords, dist=cosine)
						error_Show = (error_Show * 0.85) + (error_equip * 0.15)
					else:
						if probability1 != -1 or probability2 != -1:
							error_Show = (error_Show * 0.85) 
				#newcoding end

				# Displaying the error percentage
				positionWrite = nh1 - 245
				if (positionWrite > 10):
					cv2.putText(image_1, 'Error: {}%'.format(str(round(100*(float(error_Show)),2))), (10, positionWrite),
								cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

				# If the similarity is > 90%, take it as correct step. Otherwise incorrect step.
				positionWrite = nh1 - 170
				if (positionWrite > 10):
					if error_Show < check_rate:
						cv2.putText(image_1, "O.K.", (40, positionWrite),
									cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
						correct_frames += 1
					else:
						cv2.putText(image_1,  "K.O.", (40, positionWrite),
									cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
						
				current_frames_count += 1

				#calc corret rate at this moment
				correct_frames_ratio_list.append(round(100*correct_frames/current_frames_count, 2))


				positionWrite = nh1 - 270
				if (positionWrite > 10):
					cv2.putText(image_1, "FPS: %f" % (1.0 / (time.time() - fps_time)), (10, positionWrite),
							cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
					#ADD===
					cv2.putText(image_1, "Speed: %s" % ('{:.2f}'.format(speed_user)), (10, positionWrite-20),
							cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
					cv2.putText(image_1, "acceleration: %s" % ('{:.2f}'.format(acceleration_user)), (10, positionWrite-40),
							cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)					
					#ADD===END

				# Display the dynamic accuracy of dance as the percentage of frames that appear as correct
				if frame_counter==0:
					frame_counter = user_cam.get(cv2.CAP_PROP_FRAME_COUNT)
				positionWrite = nh1 - 220
				if (positionWrite > 10):
					cv2.putText(image_1, "Accurately done: {}%".format(str(round(100*correct_frames/frame_counter, 2))), (10, positionWrite), 
							cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

				if (output_filename!=''):
					if (combine_result):
						if h1 > w1:
							im_h=cv2.hconcat([image_2, image_1])
						else:
							im_h=cv2.vconcat([image_2, image_1])
						out.write(im_h)
					else:
						out.write(image_1)
					saved_frames +=1

				# Display both the benchmark and the user videos
#				if show_window:
#					cv2.imshow('Benchmark Video', image_2)
#					cv2.imshow('User Video', image_1)

				fps_time = time.time()
#				if cv2.waitKey(1) & 0xFF == ord('q'):
#					break
			else:
				break
		except Exception as error:
			print("An exception occurred:", error) # An exception occurred: division by zero
			pass

	benchmark_cam.release()
	user_cam.release()
	if (output_filename!=''):
		out.release()
#	cv2.destroyAllWindows()

#	print ('completed compare')
	if saved_frames > 1 and blob_connection != '' and blob_containername != '' and output_fullname !='':
		# Create a BlockBlobService object
		outputcontainer_client = ContainerClient.from_connection_string(conn_str=blob_connection, container_name=blob_containername)

		# Upload file
		with open(output_filename, "rb") as data:
			outputcontainer_client.upload_blob(name=output_fullname, data=data, overwrite=True)

	try:
		if deleted_blob != '' and output_fullname != deleted_blob:
			deletecontainer_client = ContainerClient.from_connection_string(conn_str=blob_connection, container_name=blob_containername)
			blob_clientdelete = deletecontainer_client.get_blob_client(blob=deleted_blob)
			blob_clientdelete.delete_blob()
	except Exception as error:
			print("An delete exception occurred:", error) # An exception occurred: division by zero
			pass
	

	if (benchmark_blobcontainer != ''):
		# Delete the file
		os.remove(newBenchmark_video)
		os.remove(newUser_video)
		os.remove(output_filename)

	#calc joints statistic
	result_dict = calc_joints_stat(all_2d_points_user, all_3d_world_points_user, frame_fps_user, joints_dict)
	
	matched_count = round(100*correct_frames/benchmark_count, 2)
	
	result_dict["matched"] = matched_count
	result_dict["matched_list"] = correct_frames_ratio_list
	

	if saved_frames > 1 and benchmark_count > 0:
		return json.dumps(result_dict)
	else:
		return -1

# return the dict contains joints calculation result	
def calc_joints_stat(all_2d_points_user, all_3d_world_points_user, frame_fps_user, joints_dict):

	mp_pose = mp.solutions.pose 

	result_dict = {}
	#real world distance in meter to find out BODY SIZE
	left_hip_world_pos = get_mp_position_timeseries(all_3d_world_points_user, 23)
	left_knee_world_pos = get_mp_position_timeseries(all_3d_world_points_user, 25)
	left_ankle_world_pos = get_mp_position_timeseries(all_3d_world_points_user, 27)

	right_hip_world_pos = get_mp_position_timeseries(all_3d_world_points_user, 24)
	right_knee_world_pos = get_mp_position_timeseries(all_3d_world_points_user, 26)
	right_ankle_world_pos = get_mp_position_timeseries(all_3d_world_points_user, 28)

	distance_1 = get_avg_distinance_of_two_joint(left_hip_world_pos, left_knee_world_pos)
	distance_2 = get_avg_distinance_of_two_joint(left_knee_world_pos, left_ankle_world_pos)

	distance_3 = get_avg_distinance_of_two_joint(right_hip_world_pos, right_knee_world_pos)
	distance_4 = get_avg_distinance_of_two_joint(right_knee_world_pos, right_ankle_world_pos)


	left_shoulder_world_pos = get_mp_position_timeseries(all_3d_world_points_user, 11)
	left_elbow_world_pos = get_mp_position_timeseries(all_3d_world_points_user, 13)
	left_wrist_world_pos = get_mp_position_timeseries(all_3d_world_points_user, 15)

	right_shoulder_world_pos = get_mp_position_timeseries(all_3d_world_points_user, 12)
	right_elbow_world_pos = get_mp_position_timeseries(all_3d_world_points_user, 14)
	right_wrist_world_pos = get_mp_position_timeseries(all_3d_world_points_user, 16)

	distance_5 = get_avg_distinance_of_two_joint(left_shoulder_world_pos, left_elbow_world_pos)
	distance_6 = get_avg_distinance_of_two_joint(left_elbow_world_pos, left_wrist_world_pos)

	distance_7 = get_avg_distinance_of_two_joint(right_shoulder_world_pos, right_elbow_world_pos)
	distance_8 = get_avg_distinance_of_two_joint(right_elbow_world_pos, right_wrist_world_pos)

	#print("===================distance of left hip and left knee:")
	#print(distiance_1)

	#print("===================distance of left knee and left ankle:")
	#print(distiance_2)

	result_dict["left_hip_keen_distance"] = distance_1
	result_dict["left_keen_ankle_distance"] = distance_2
	result_dict["right_hip_keen_distance"] = distance_3
	result_dict["right_keen_ankle_distance"] = distance_4
	result_dict["left_shoulder_elbow_distance"] = distance_5
	result_dict["left_elbow_wrist_distance"] = distance_6
	result_dict["right_shoulder_elbow_distance"] = distance_7
	result_dict["right_elbow_wrist_distance"] = distance_8

	#calc speed
	joint1_pos = get_mp_position_timeseries(all_3d_world_points_user, joints_dict["joint1"])
	avg_speed = avg_speed_in_1_std(joint1_pos, 1/frame_fps_user)
	result_dict["joint1_avg_speed"] = avg_speed

	joint1_pos = get_mp_position_timeseries(all_3d_world_points_user, joints_dict["joint2"])
	avg_speed = avg_speed_in_1_std(joint1_pos, 1/frame_fps_user)
	result_dict["joint2_avg_speed"] = avg_speed

	joint1_pos = get_mp_position_timeseries(all_3d_world_points_user, joints_dict["joint3"])
	avg_speed = avg_speed_in_1_std(joint1_pos, 1/frame_fps_user)
	result_dict["joint3_avg_speed"] = avg_speed

	joint1_pos = get_mp_position_timeseries(all_3d_world_points_user, joints_dict["joint4"])
	avg_speed = avg_speed_in_1_std(joint1_pos, 1/frame_fps_user)
	result_dict["joint4_avg_speed"] = avg_speed


	# Calculate  Range of Motion ROM for Upper and Lower Body to reflect flexibility

	# Variables to store angles
	shoulder_angles =list()
	elbow_angles = list()
	hip_angles = list()
	knee_angles = list()
	
	left_shoulder_pos = get_mp_position_timeseries(all_2d_points_user, mp_pose.PoseLandmark.LEFT_SHOULDER)
	left_elbow_pos = get_mp_position_timeseries(all_2d_points_user, mp_pose.PoseLandmark.LEFT_ELBOW)
	left_wrist_pos = get_mp_position_timeseries(all_2d_points_user, mp_pose.PoseLandmark.LEFT_WRIST)
	left_hip_pos = get_mp_position_timeseries(all_2d_points_user, mp_pose.PoseLandmark.LEFT_HIP)
	left_knee_pos = get_mp_position_timeseries(all_2d_points_user, mp_pose.PoseLandmark.LEFT_KNEE)
	left_ankle_pos = get_mp_position_timeseries(all_2d_points_user, mp_pose.PoseLandmark.LEFT_ANKLE)

	# Calculate shoulder and elbow angles
	for indx, shoulder in enumerate(left_shoulder_pos):
		if (shoulder is not None and left_elbow_pos[indx] is not None and left_wrist_pos[indx] is not None) :
			elbow_angle = calculate_angle(shoulder, left_elbow_pos[indx], left_wrist_pos[indx])
			shoulder_angle = calculate_angle(left_elbow_pos[indx] , shoulder, left_wrist_pos[indx])
			
			shoulder_angles.append(shoulder_angle)
			elbow_angles.append(elbow_angle)

	# Calculate hip and knee angles
	for indx, hip in enumerate(left_hip_pos):
		if (hip is not None and left_knee_pos[indx] is not None and left_ankle_pos[indx] is not None) :
			knee_angle = calculate_angle(hip, left_knee_pos[indx], left_ankle_pos[indx])
			hip_angle = calculate_angle(left_knee_pos[indx] , hip, left_ankle_pos[indx])
			
			hip_angles.append(hip_angle)
			knee_angles.append(knee_angle)


	shoulder_rom = max(shoulder_angles) - min(shoulder_angles)
	elbow_rom = max(elbow_angles) - min(elbow_angles)
	hip_rom = max(hip_angles) - min(hip_angles)
	knee_rom = max(knee_angles) - min(knee_angles)

	#print(f'Range of Motion for Shoulder: {shoulder_rom:.2f} degrees')
	#print(f'Range of Motion for Elbow: {elbow_rom:.2f} degrees')
	#print(f'Range of Motion for Hip: {hip_rom:.2f} degrees')
	#print(f'Range of Motion for Knee: {knee_rom:.2f} degrees')

	result_dict["shoulder_rom"] = shoulder_rom
	result_dict["elbow_rom"] = elbow_rom
	result_dict["hip_rom"] = hip_rom
	result_dict["knee_rom"] = knee_rom

# calc accleration of joints_dict["joint1"]
	joint1_pos_for_acc = get_mp_position_timeseries(all_3d_world_points_user, joints_dict["joint1"])
	joint1_acc_result = calculate_acceleration_from_joints(joint1_pos_for_acc, 1/frame_fps_user)
	#print(f'joint1_acc_result')
	#print(joint1_acc_result)
	result_dict["joint1_max_acceleration"] = max(joint1_acc_result)

	result_dict["joint1_name"] = get_joint_name(joints_dict["joint1"])
	result_dict["joint2_name"] = get_joint_name(joints_dict["joint2"])
	result_dict["joint3_name"] = get_joint_name(joints_dict["joint3"])
	result_dict["joint4_name"] = get_joint_name(joints_dict["joint4"])
	
	return result_dict
