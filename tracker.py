from torch.functional import Tensor
from api import Detector
import numpy as np
import cv2
from mylib.centroidtracker import CentroidTracker
from PIL import Image
from PIL import ImageFile
from mylib.centroidtracker import CentroidTracker
from mylib.trackableobject import TrackableObject
from imutils.video import VideoStream
from imutils.video import FPS
#from imutils import resize
from mylib.mailer import Mailer
from mylib import config, thread
import time, schedule, csv
import argparse, imutils
import time, dlib, cv2, datetime
from itertools import zip_longest
from Lineiterator import createLineIterator
from limitexceed import check_exceed
from get_requests import send_req
import pandas as pd
import datetime
from os.path import exists
from excel_appender import append_df_to_excel
from excel_data_converter import create_summary, data_converter
from mylib.config import x1,y1,x2,y2, vertical_direction, enter_direction,hi,wi


def run():
    ap = argparse.ArgumentParser()
    ap.add_argument("-cam", "--camera", required=True,type=str,help="summary camera name")
    args = vars(ap.parse_args())
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    def cv2pil(imgCV):
        imgCV_RGB = cv2.cvtColor(imgCV, cv2.COLOR_BGR2RGB)
        imgPIL = Image.fromarray(imgCV_RGB)
        return imgPIL  

    detector = Detector(model_name='rapid',
                        weights_path='./weights/pL1_MWHB1024_Mar11_4000.ckpt',use_cuda=False)

    W = None
    H = None

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

    cap = VideoStream(src=config.url).start()

    if config.Thread:
            vs = thread.ThreadingClass(config.url)


    while True:
        # フレームを取得
        frame = cap.read()

        try:
            frame = imutils.resize(frame, width = 500)
        except AttributeError:
            print(frame)
            raise AttributeError
            
        try:

            if W is None or H is None:
                (H, W) = frame.shape[:2]
                print(f"Frame height  is : {H}, frame width is : {W}")
        except AttributeError:
            print('(H, W) = frame.shape[:2] error')
            raise AttributeError  

        #if ret == False:
        # break
        #pil_img1=cv2pil(frame)
        #npobj=detector(model_name='rapid',)
        np_img, detections = detector.detect_one(pil_img=cv2pil(frame),
                        input_size=512, conf_thres=0.3, return_img=True
                        )
        detections=Tensor.tolist(detections)
        objects = ct.update(detections)

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
                    if len(y) >= 30:
                        # sum  of xi - mean(xi-1)
                        #try
                            #direction_all=[]
                        for index,i in enumerate(y[-31:]):
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
                    if len(y) >= 30:
                        # sum  of xi - mean(xi-1)
                        #try
                            #direction_all=[]
                        for index,i in enumerate(y[-31:]):
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
                    if centroid[0] < iterlist[0][0] or centroid[0] > iterlist[-1][0]:
                        pass
                    elif m == 1000000001 and (centroid[1] < iterlist[0][1] or centroid[1] > iterlist[-1][1]):
                        pass
                    else:
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

        if enter_direction == 'down' or enter_direction == 'right':
            info = [
            ("Exit", totalUp),
            ("Enter", totalDown),
            ]
        else:
            info = [
            ("Exit", totalDown),
            ("Enter", totalUp),        
            ]

        info2 = [
        ("Total people inside", x)
        ]
            #print(peoplechangelist)
        if config.people_change == True:
                        #if len(peoplechangelist) > 0:
            peoplechangelist.append(x)
                                #print(peoplechangelist)
        
    #new_image = np.array(np_img, dtype=np.uint8)
    #np_img2 = cv2.cvtColor(new_image, cv2.COLOR_RGB2BGR)
    # フレームを表示
    #cv2.imshow("Frame", np_img2)
        
        # フレームを表示
        new_image = np.array(np_img, dtype=np.uint8)
        np_img2 = cv2.cvtColor(new_image, cv2.COLOR_RGB2BGR)

            # draw both the ID of the object and the centroid of the
                # object on the output frame
        try:
            assert objectID
            assert centroid
            text = "ID {}".format(objectID)
            cv2.putText(np_img2, text, (centroid[0] - 10, centroid[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            cv2.circle(np_img2, (centroid[0], centroid[1]), 4, (255, 255, 255), -1)
        except (AssertionError,ValueError,NameError):
            pass
        try:
            if int(info2[0][1][0]) >= config.Threshold:
                cv2.putText(frame, "-ALERT: People limit exceeded-", (10, frame.shape[0] - 80),
                    cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 2)
                if config.ALERT:
                    print("[INFO] Sending email alert..")
                    Mailer().send(config.MAIL)
                    print("[INFO] Alert sent")
        except IndexError:
            pass

        for (i, (k, v)) in enumerate(info):
            text = "{}: {}".format(k, v)
            cv2.putText(np_img2, text, (30, hi - ((i * 20) + 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        for (i, (k, v)) in enumerate(info2):
            text = "{}: {}".format(k, v)
            cv2.putText(np_img2, text, (265, hi - ((i * 20) + 60)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
        # Initiate a simple log to save data at end of the day
        if config.Log:

            try:
                timeinxmins
            except NameError:
                timeinxmins=datetime.datetime.now() + config.timedel

            if datetime.datetime.now() >= timeinxmins:
                #data={'櫃位地點':config.cam_place,'People Enter':info[1][1],'People Exit':info[0][1],'Current People Inside':info2[0][1],'Date':datetime.datetime.now()}
                #df=pd.DataFrame(data=data)
                timeinxmins=datetime.datetime.now() + config.timedel
                cam_place=str(args["camera"])
                excel_name=f"./summary/{cam_place} summary.xlsx"
                if exists(excel_name):
                    #with pd.ExcelWriter(excel_name,mode='a')  as writer:
                    #append_df_to_excel(excel_name, df,header=None, index=False)
                    data_converter(info[1][1],info[0][1],excel_name)  
                else:
                    create_summary(info[1][1],info[0][1],excel_name)
                print('summary exported!')
            


        cv2.line(np_img2, (int(round(x1)), int(round(y1))), (int(round(x2)), int(round(y2))), (0, 0, 255), 3)

        cv2.imshow("Frame", np_img2)
        # qキーが押されたら途中終了
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
#print('finished')

if config.Scheduler:
	##Runs for every 1 second
	schedule.every(1).seconds.do(run)
	##Runs at every day (9:00 am). You can change it.
	#schedule.every().day.at("9:00").do(run)

	while 1:
		schedule.run_pending()
else:
    run()