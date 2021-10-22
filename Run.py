from mylib.centroidtracker import CentroidTracker
from mylib.trackableobject import TrackableObject
from imutils.video import VideoStream
from imutils.video import FPS
#from imutils import resize
from mylib.mailer import Mailer
from mylib import config, thread
import time, schedule, csv
import numpy as np
import argparse, imutils
import time, dlib, cv2, datetime
from itertools import zip_longest
from Lineiterator import createLineIterator
from limitexceed import check_exceed
from get_requests import send_req

t0 = time.time()

def run():

	# construct the argument parse and parse the arguments
	ap = argparse.ArgumentParser()
	ap.add_argument("-p", "--prototxt", required=False,
		help="path to Caffe 'deploy' prototxt file")
	ap.add_argument("-m", "--model", required=True,
		help="path to Caffe pre-trained model")
	ap.add_argument("-i", "--input", type=str,
		help="path to optional input video file")
	ap.add_argument("-o", "--output", type=str,
		help="path to optional output video file")
	# confidence default 0.4
	ap.add_argument("-c", "--confidence", type=float, default=0.4,
		help="minimum probability to filter weak detections")
	ap.add_argument("-s", "--skip-frames", type=int, default=30,
		help="# of skip frames between detections")
	args = vars(ap.parse_args())

	# initialize the list of class labels MobileNet SSD was trained to
	# detect
	CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
		"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
		"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
		"sofa", "train", "tvmonitor"]

	# load our serialized model from disk
	net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

	# if a video path was not supplied, grab a reference to the ip camera
	if not args.get("input", False):
		print("[INFO] Starting the live stream..")
		vs = VideoStream(src=config.url).start()
		#time.sleep(2.0)



	# otherwise, grab a reference to the video file
	else:
		print("[INFO] Starting the video..")
		vs = cv2.VideoCapture(args["input"])

	# initialize the video writer (we'll instantiate later if need be)
	writer = None

	# initialize the frame dimensions (we'll set them as soon as we read
	# the first frame from the video)
	W = None
	H = None
	############ Win's modification parameters ##################
	# height and width of frame
	# start running the video first, it's height and width will be printed in the cmd then type it in here
	# hi = height and wi = width
	#hi=373
	#wi=500
	hi=373
	wi=500
	#specify points here the line will be draw from x1,y1 to x2,y2
	#note that when hi = 0, it will be at the top and when hi = max it will be at the bottom
	x1=0
	y1=hi  // 2
	x2=wi
	y2=hi  // 2
	#x1=wi  * 0.50
	#y1=hi
	#x2=wi * 0.75
	#y2=0
	# set vertical_direction to 1 for catagorize people in a vertical movement
	# set to 0 for catagorize people by horizon movement
	# if the line has a negative or positive slope, set to 0 or 1 will do
	#0 will use movement up/downward for calculations, 1 will use movement left/right for calculations
	vertical_direction = 1
	#Specify enter side 
	# if vertical_direction = 1 , use "up" for enter up and "down" for enter down
	# if vertical_direction = 0 , use "left" for enter left side and "right" for enter right side
	enter_direction = 'down'
	#time
	if config.five_mins == True:
		now=datetime.datetime.now()
		fivemin= now+datetime.timedelta(0,300)
	if config.people_change == True:
		peoplechangelist= []
	###################################
	try:
		m = ((-1*y2)-y1)/((x2)-x1)
	except:
		m = 1000000001
	print(m)
	# m = (y2-y1)/(x2-x1)
	# 0,0 -w // 2, -hi
	#print(m)
	iterlist=createLineIterator(np.array([int(round(x1)), int(round(y1))]),np.array([int(round(x2)), int(round(y2))]))
	#cv2.line(frame, (0, int(round(0))), (W // 2, int(round(H))), (0, 0, 0), 3)
	#iterlist=createLineIterator(np.array([0, round(hi * 0.8)]),np.array([wi, round(hi * 0.80)]))
	#iterlist= [(x,hi-y) for (x,y) in iterlist]
	#print(iterlist)
	#print(iterlist)
	# instantiate our centroid tracker, then initialize a list to store
	# each of our dlib correlation trackers, followed by a dictionary to
	# map each unique object ID to a TrackableObject
	ct = CentroidTracker(maxDisappeared=40, maxDistance=50)
	trackers = []
	trackableObjects = {}

	# initialize the total number of frames processed thus far, along
	# with the total number of objects that have moved either up or down
	totalFrames = 0
	x = []
	#if enter_direction == 'down':
	totalDown = 0
	totalUp = 0
	empty=[]
	empty1=[]

	# start the frames per second throughput estimator
	fps = FPS().start()

	if config.Thread:
		vs = thread.ThreadingClass(config.url)

	# loop over frames from the video stream
	while True:
		# grab the next frame and handle if we are reading from either
		# VideoCapture or VideoStream
		frame = vs.read()
				
		frame = frame[1] if args.get("input", False) else frame

		# if we are viewing a video and we did not grab a frame then we
		# have reached the end of the video
		if args["input"] is not None and frame is None:
			break

		# resize the frame to have a maximum width of 500 pixels (the
		# less data we have, the faster we can process it), then convert
		# the frame from BGR to RGB for dlib
		try:
			frame = imutils.resize(frame, width = 500)
		except AttributeError:
			print(frame)
			raise AttributeError


		rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

		# if the frame dimensions are empty, set them
		try:

			if W is None or H is None:
				(H, W) = frame.shape[:2]
				print(f"Frame height  is : {H}, frame width is : {W}")
		except AttributeError:
			print('(H, W) = frame.shape[:2] error')
			raise AttributeError

		# if we are supposed to be writing a video to disk, initialize
		# the writer
		if args["output"] is not None and writer is None:
			fourcc = cv2.VideoWriter_fourcc(*"mp4v")
			writer = cv2.VideoWriter(args["output"], fourcc, 30,
				(W, H), True)

		# initialize the current status along with our list of bounding
		# box rectangles returned by either (1) our object detector or
		# (2) the correlation trackers
		status = "Waiting"
		rects = []

		# send requests
		if config.five_mins == True:
			if datetime.datetime.now() >= fivemin:		
				enterp=info[1][1]
				exitp=info[0][1]
				send_req(enterp,exitp)
				now = datetime.datetime.now()
				fivemin = now + datetime.timedelta(0,300)
		if config.people_change == True:
			if len(peoplechangelist) >= 2:
				if peoplechangelist[-1] != peoplechangelist[-2]:
					enterp=info[1][1]
					exitp=info[0][1]
					print(peoplechangelist)
					send_req(enterp,exitp)
			if len(peoplechangelist) > 2:
				del peoplechangelist[:-2]

		# check to see if we should run a more computationally expensive
		# object detection method to aid our tracker
		if totalFrames % args["skip_frames"] == 0:
			# set the status and initialize our new set of object trackers
			status = "Detecting"
			trackers = []

			# convert the frame to a blob and pass the blob through the
			# network and obtain the detections
			blob = cv2.dnn.blobFromImage(frame, 0.007843, (W, H), 127.5)
			net.setInput(blob)
			detections = net.forward()

			# loop over the detections
			for i in np.arange(0, detections.shape[2]):
				# extract the confidence (i.e., probability) associated
				# with the prediction
				confidence = detections[0, 0, i, 2]

				# filter out weak detections by requiring a minimum
				# confidence
				if confidence > args["confidence"]:
					# extract the index of the class label from the
					# detections list
					idx = int(detections[0, 0, i, 1])

					# if the class label is not a person, ignore it
					if CLASSES[idx] != "person":
						continue

					# compute the (x, y)-coordinates of the bounding box
					# for the object
					box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
					(startX, startY, endX, endY) = box.astype("int")


					# construct a dlib rectangle object from the bounding
					# box coordinates and then start the dlib correlation
					# tracker
					tracker = dlib.correlation_tracker()
					rect = dlib.rectangle(startX, startY, endX, endY)
					tracker.start_track(rgb, rect)

					# add the tracker to our list of trackers so we can
					# utilize it during skip frames
					trackers.append(tracker)

		# otherwise, we should utilize our object *trackers* rather than
		# object *detectors* to obtain a higher frame processing throughput
		else:
			# loop over the trackers
			for tracker in trackers:
				# set the status of our system to be 'tracking' rather
				# than 'waiting' or 'detecting'
				status = "Tracking"

				# update the tracker and grab the updated position
				tracker.update(rgb)
				pos = tracker.get_position()

				# unpack the position object
				startX = int(pos.left())
				startY = int(pos.top())
				endX = int(pos.right())
				endY = int(pos.bottom())

				# add the bounding box coordinates to the rectangles list
				rects.append((startX, startY, endX, endY))

		# draw a horizontal line in the center of the frame -- once an
		# object crosses this line we will determine whether they were
		# moving 'up' or 'down'
		#cv2.line(frame, (0, H // 2), (W, H // 2), (0, 0, 0), 3)
		#cv2.line(frame, (0,220), (W,120), (0, 0, 0), 3)
		#cv2.line(frame, (0, int(round(H * 0.50))), (W, int(round(H * 0.66))), (0, 0, 0), 3)
		#cv2.line(frame, (0, int(round(H * 0.80))), (W, int(round(H * 0.80))), (0, 0, 0), 3)
		cv2.line(frame, (int(round(x1)), int(round(y1))), (int(round(x2)), int(round(y2))), (0, 0, 255), 3)
		#iterlist=createLineIterator(np.array([0, round(H * 0.50)]),np.array([W, round(H * 0.66)]),frame)
		#print(len(iterlist))
		cv2.putText(frame, "-Prediction border - Entrance-", (10, H - ((i * 20) + 200)),
			cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

		# use the centroid tracker to associate the (1) old object
		# centroids with (2) the newly computed object centroids
		objects = ct.update(rects)

		# loop over the tracked objects
		for (objectID, centroid) in objects.items():
			# check to see if a trackable object exists for the current
			# object ID
			to = trackableObjects.get(objectID, None)

			# if there is no existing trackable object, create one
			if to is None:
				to = TrackableObject(objectID, centroid)

			# otherwise, there is a trackable object so we can utilize it
			# to determine direction
			else:
				# the difference between the y-coordinate of the *current*
				# centroid and the mean of *previous* centroids will tell
				# us in which direction the object is moving (negative for
				# 'up' and positive for 'down')
				if vertical_direction == 1:
					y = [c[1] for c in to.centroids]
					#print(y)
					#direction = centroid[1] - np.mean(y)
					direction =  0
					to.centroids.append(centroid)
					#print(to.centroids)
					direction_all=[]
					if len(y) >= 40:
						# sum  of xi - mean(xi-1)
						#try
							#direction_all=[]
						for index,i in enumerate(y[-41:]):
							prev_mean= np.mean(y[:index+1])
							direc= i - prev_mean
							direction_all.append(direc)
						if all([x > 0 for x in direction_all]):
							direction = 1
						elif all([x < 0 for x in direction_all]):    
							direction = -1
						else:
							direction = 0
						#except
				else:
					y = [c[0] for c in to.centroids]
					#print(y)
					#direction = centroid[1] - np.mean(y)
					direction =  0
					to.centroids.append(centroid)
					#print(to.centroids)
					direction_all=[]
					if len(y) >= 40:
						# sum  of xi - mean(xi-1)
						#try
							#direction_all=[]
						for index,i in enumerate(y[-41:]):
							prev_mean= np.mean(y[:index+1])
							direc= i - prev_mean
							direction_all.append(direc)
						if all([x > 0 for x in direction_all]):
							# right
							direction = 1
						elif all([x < 0 for x in direction_all]):
							#left    
							direction = -1
						else:
							direction = 0

				# check to see if the object has been counted or not
				if not to.counted:
					if m < 0 and vertical_direction == 1:
						#if the direction is negative (indicating the object
						#is moving up) AND the centroid is above the center
						#line, count the object
						#H is between 0 and 500 the over the value the upper it will be, the higher the value, the lower it will be.
						#if direction < 0 and centroid[1] < int(round(H * 0.66)):
						#print(str(centroid))
						if direction < 0:
							for i in iterlist:
								if centroid[0] > i[0] and centroid[1] < i[1]:
							
									totalUp += 1
									empty.append(totalUp)
									to.counted = True
									print('ID '+ str(to.objectID) + ' going up' + ' direction : ' + str(direction) + ' centroid : ' + str(centroid) + ' pixcel compared to : ' + str(i[0]) + ' ' + str(i[1]))
									if enter_direction == 'up':
										check_exceed(x,frame)
									break

						# if the direction is positive (indicating the object
						# is moving down) AND the centroid is below the
						# center line, count the object
						#elif direction > 0 and centroid[1] > int(round(H * 0.66)):
						elif direction > 0:
							for i in iterlist:
								if centroid[0] < i[0] and centroid[1] > i[1]:
									totalDown += 1
									empty1.append(totalDown)
									to.counted = True
									print('ID '+ str(to.objectID) + ' going down' + ' direction : ' + str(direction) + ' centroid : ' + str(centroid) + ' pixcel compared to : ' + str(i[0]) + ' ' + str(i[1]))
									if enter_direction == 'down':
										check_exceed(x,frame)
									break
									#print(empty1[-1])
									# if the people limit exceeds over threshold, send an email alert
					elif m == 0 and vertical_direction == 1:
						
						if direction < 0:
							for i in iterlist:
								if centroid[1] < i[1]:
							
									totalUp += 1
									empty.append(totalUp)
									to.counted = True
									print('ID '+ str(to.objectID) + ' going up' + ' direction : ' + str(direction) + ' centroid : ' + str(centroid) + ' pixcel compared to : ' + str(i[0]) + ' ' + str(i[1]))
									if enter_direction == 'up':
										check_exceed(x,frame)
									break

						
						elif direction > 0:
							for i in iterlist:
								if centroid[1] > i[1]:
									totalDown += 1
									empty1.append(totalDown)
									to.counted = True
									print('ID '+ str(to.objectID) + ' going down' + ' direction : ' + str(direction) + ' centroid : ' + str(centroid) + ' pixcel compared to : ' + str(i[0]) + ' ' + str(i[1]))
									if enter_direction == 'down':
										check_exceed(x,frame)
									break
									
					elif 0 < m < 1000000000 and vertical_direction == 1:
						
						if direction < 0:
							for i in iterlist:
								if centroid[0] < i[0] and centroid[1] < i[1]:
							
									totalUp += 1
									empty.append(totalUp)
									to.counted = True
									print('ID '+ str(to.objectID) + ' going up' + ' direction : ' + str(direction) + ' centroid : ' + str(centroid) + ' pixcel compared to : ' + str(i[0]) + ' ' + str(i[1]))
									if enter_direction == 'up':
										check_exceed(x,frame)
									break

						
						elif direction > 0:
							for i in iterlist:
								if centroid[0] > i[0] and centroid[1] > i[1]:
									totalDown += 1
									empty1.append(totalDown)
									to.counted = True
									print('ID '+ str(to.objectID) + ' going down' + ' direction : ' + str(direction) + ' centroid : ' + str(centroid) + ' pixcel compared to : ' + str(i[0]) + ' ' + str(i[1]))
									if enter_direction == 'down':
										check_exceed(x,frame)
									break
									
					elif m < 0 and vertical_direction == 0:
						
						# if the direction is negative (indicating the object
						# is moving LEFT) AND the centroid is on the left side
						# line, count the object
					
						
						if direction < 0:
							for i in iterlist:
								if centroid[0] < i[0] and centroid[1] > i[1]:
							
									totalUp += 1
									empty.append(totalUp)
									to.counted = True
									print('ID '+ str(to.objectID) + ' going left' + ' direction : ' + str(direction) + ' centroid : ' + str(centroid) + ' pixcel compared to : ' + str(i[0]) + ' ' + str(i[1]))
									if enter_direction == 'left':
										check_exceed(x,frame)
									break

						# if the direction is positive (indicating the object
						# is moving RIGHT) AND the centroid is on the the side
						#  line, count the object
						elif direction > 0:
							for i in iterlist:
								if centroid[0] > i[0] and centroid[1] < i[1]:
									totalDown += 1
									empty1.append(totalDown)
									to.counted = True
									print('ID '+ str(to.objectID) + ' going right' + ' direction : ' + str(direction) + ' centroid : ' + str(centroid) + ' pixcel compared to : ' + str(i[0]) + ' ' + str(i[1]))
									if enter_direction == 'right':
										check_exceed(x,frame)
									break
									
					elif m >= 1000000000 and vertical_direction == 0:
						# m is infinite/ vertical line
						if direction < 0:
							for i in iterlist:
								if centroid[0] < i[0]:
							
									totalUp += 1
									empty.append(totalUp)
									to.counted = True
									print('ID '+ str(to.objectID) + ' going left' + ' direction : ' + str(direction) + ' centroid : ' + str(centroid) + ' pixcel compared to : ' + str(i[0]) + ' ' + str(i[1]))
									if enter_direction == 'left':
										check_exceed(x,frame)
									break

						
						elif direction > 0:
							for i in iterlist:
								if centroid[0] > i[0]:
									totalDown += 1
									empty1.append(totalDown)
									to.counted = True
									print('ID '+ str(to.objectID) + ' going right' + ' direction : ' + str(direction) + ' centroid : ' + str(centroid) + ' pixcel compared to : ' + str(i[0]) + ' ' + str(i[1]))
									if enter_direction == 'right':
										check_exceed(x,frame)
									break
									
					elif 0 < m < 1000000000 and vertical_direction == 0:
						
						if direction < 0:
							for i in iterlist:
								if centroid[0] < i[0] and centroid[1] < i[1]:
							
									totalUp += 1
									empty.append(totalUp)
									to.counted = True
									print('ID '+ str(to.objectID) + ' going left' + ' direction : ' + str(direction) + ' centroid : ' + str(centroid) + ' pixcel compared to : ' + str(i[0]) + ' ' + str(i[1]))
									if enter_direction == 'left':
										check_exceed(x,frame)
									break

						
						elif direction > 0:
							for i in iterlist:
								if centroid[0] > i[0] and centroid[1] > i[1]:
									totalDown += 1
									empty1.append(totalDown)
									to.counted = True
									print('ID '+ str(to.objectID) + ' going right' + ' direction : ' + str(direction) + ' centroid : ' + str(centroid) + ' pixcel compared to : ' + str(i[0]) + ' ' + str(i[1]))
									if enter_direction == 'right':
										check_exceed(x,frame)
									break
										
					x = []
					# compute the sum of total people inside
					if enter_direction == 'down' or enter_direction == 'right':
						x.append(len(empty1)-len(empty))
					else:
						x.append(len(empty)-len(empty1))
					#print("Total people inside:", x)


			# store the trackable object in our dictionary
			trackableObjects[objectID] = to

			# draw both the ID of the object and the centroid of the
			# object on the output frame
			text = "ID {}".format(objectID)
			cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
				cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
			cv2.circle(frame, (centroid[0], centroid[1]), 4, (255, 255, 255), -1)

		# construct a tuple of information we will be displaying on the
		if enter_direction == 'down' or enter_direction == 'right':
			info = [
			("Exit", totalUp),
			("Enter", totalDown),
			("Status", status),
			]
		else:
			info = [
			("Exit", totalDown),
			("Enter", totalUp),
			("Status", status),
			]

		info2 = [
		("Total people inside", x)
		]
		#print(peoplechangelist)
		if config.people_change == True:
                        #if len(peoplechangelist) > 0:
			peoplechangelist.append(x)
                                #print(peoplechangelist)
                
                # Display the output
		for (i, (k, v)) in enumerate(info):
			text = "{}: {}".format(k, v)
			cv2.putText(frame, text, (10, H - ((i * 20) + 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

		for (i, (k, v)) in enumerate(info2):
			text = "{}: {}".format(k, v)
			cv2.putText(frame, text, (265, H - ((i * 20) + 60)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

		# Initiate a simple log to save data at end of the day
		if config.Log:
			datetimee = [datetime.datetime.now()]
			d = [datetimee, empty1, empty, x]
			export_data = zip_longest(*d, fillvalue = '')

			with open('Log.csv', 'w', newline='') as myfile:
				wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
				wr.writerow(("End Time", "In", "Out", "Total Inside"))
				wr.writerows(export_data)
				
		# check to see if we should write the frame to disk
		if writer is not None:
			writer.write(frame)

		# show the output frame
		cv2.imshow("Real-Time Monitoring/Analysis Window", frame)
		key = cv2.waitKey(1) & 0xFF

		# if the `q` key was pressed, break from the loop
		if key == ord("q"):
			break

		# increment the total number of frames processed thus far and
		# then update the FPS counter
		totalFrames += 1
		fps.update()

		if config.Timer:
			# Automatic timer to stop the live stream. Set to 8 hours (28800s).
			t1 = time.time()
			num_seconds=(t1-t0)
			if num_seconds > 28800:
				break

	# stop the timer and display FPS information
	fps.stop()
	print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
	print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))


	# # if we are not using a video file, stop the camera video stream
	# if not args.get("input", False):
	# 	vs.stop()
	#
	# # otherwise, release the video file pointer
	# else:
	# 	vs.release()
	
	# issue 15
	if config.Thread:
		vs.release()

	# close any open windows
	cv2.destroyAllWindows()


##learn more about different schedules here: https://pypi.org/project/schedule/
if config.Scheduler:
	##Runs for every 1 second
	#schedule.every(1).seconds.do(run)
	##Runs at every day (9:00 am). You can change it.
	schedule.every().day.at("9:00").do(run)

	while 1:
		schedule.run_pending()

else:
	run()
