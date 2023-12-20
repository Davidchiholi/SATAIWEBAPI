import cv2, time
import os
import random
import pose_module as pm
##new code begin
import numpy as np
import mediapipe as mp
##new code end
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

def compare_positions(benchmark_video, user_video, benchmark_blobcontainer, user_blobcontainer, output_filename, output_fullname, blob_containername, check_rate, blob_connection, sport, show_window, combine_result, deleted_blob, model, equip, model1, equip1, model2, equip2):
	if (benchmark_video=='' or user_video==''):
		return 0

	benchmarkcontainer_client = ContainerClient.from_connection_string(conn_str=blob_connection, container_name=benchmark_blobcontainer)
	newBenchmark_video = 'V'+str(time.strftime("%Y%m%d%H%M%S"))+str(random.randint(0,199999999)) + os.path.basename(benchmark_video)  
	with open(file=newBenchmark_video, mode="wb") as download_filebm:
		download_filebm.write(benchmarkcontainer_client.download_blob(benchmark_video).readall())
	
	usercontainer_client = ContainerClient.from_connection_string(conn_str=blob_connection, container_name=user_blobcontainer)
	newUser_video = 'V'+str(time.strftime("%Y%m%d%H%M%S"))+str(random.randint(0,199999999)) + os.path.basename(user_video)  
	with open(file=newUser_video, mode="wb") as download_fileur:
		download_fileur.write(usercontainer_client.download_blob(user_video).readall())

	benchmark_cam = cv2.VideoCapture(newBenchmark_video)
	user_cam = cv2.VideoCapture(newUser_video) 
	w2 = int(benchmark_cam.get(cv2.CAP_PROP_FRAME_WIDTH))
	h2 = int(benchmark_cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
	w1 = int(user_cam.get(cv2.CAP_PROP_FRAME_WIDTH))
	h1 = int(user_cam.get(cv2.CAP_PROP_FRAME_HEIGHT))
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
	else:
		base_options = python.BaseOptions(model_asset_path='efficientdet.tflite')
	options = vision.ObjectDetectorOptions(base_options=base_options,
									score_threshold=0.5)
	mp_object_detection = vision.ObjectDetector.create_from_options(options)
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
			image_2 = cv2.resize(image_2, (nw2,nh2))
			image_2F = cv2.flip(image_2, 1)
			image_2 = detector_2.findPose(image_2)
			image_2F = detector_2F.findPose(image_2F)
			lmList_benchmark = detector_2.findPosition(image_2)
			lmList_benchmark_F = detector_2F.findPosition(image_2F)
			del lmList_benchmark[1:11]
			del lmList_benchmark_F[1:11]

			frame_counter += 1

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
				positionWrite = nh1 - 270
				if (positionWrite > 10):
					cv2.putText(image_1, "FPS: %f" % (1.0 / (time.time() - fps_time)), (10, positionWrite),
							cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

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
	
	# Delete the file
	os.remove(newBenchmark_video)
	os.remove(newUser_video)
	os.remove(output_filename)
	if saved_frames > 1 and benchmark_count > 0:
		return round(100*correct_frames/benchmark_count, 2)
	else:
		return -1