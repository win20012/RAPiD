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
import schedule
import argparse, imutils
import cv2, datetime

from Lineiterator import createLineIterator
from limitexceed import check_exceed
from get_requests import send_req
import pandas as pd
import datetime
from os.path import exists
from excel_appender import append_df_to_excel
from excel_data_converter import create_summary, data_converter
from mylib.config import x1,y1,x2,y2, vertical_direction, enter_direction,hi,wi
from multiprocessing import Queue, Process
import queue

class variable:
    def __init__(self):
        self.ct = CentroidTracker(maxDisappeared=40, maxDistance=50)
        self.trackers=[]
        self.trackableObjects={}
        self.totalFrames=0
        self.x=[]
        self.totalDown=0
        self.totalUp=0
        self.empty=[]
        self.empty1=[]
        self.cap=cv2.VideoCapture(config.url)
        self.H = None
        self.W= None
        self.fivemin= datetime.datetime.now()+datetime.timedelta(0,300)
        self.writer = None
        self.do_malier = 1

var=variable() 


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

cap = VideoStream(src=config.url).start()

if config.Thread:
        vs = thread.ThreadingClass(config.url)



# フレームを取得

def capture(q):
    while True:
        
        ret, frame = var.cap.read() # read the frames and ---
        if not ret:
            break
        if not q.empty():
            try:
                q.get_nowait()
            except queue.Empty:
                pass
        q.put(frame)
        #frame = vs.read()

def tracker_peo(q):
    while True:
        frame = q.get()
        frame = frame[1] if args.get("input", False) else frame
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
        objects = var.ct.update(detections)

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
            to = var.trackableObjects.get(objectID, None)

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
                                
                                        var.totalUp += 1
                                        var.empty.append(var.totalUp)
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
                                        var.totalDown += 1
                                        var.empty1.append(var.totalDown)
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
                                
                                        var.totalUp += 1
                                        var.empty.append(var.totalUp)
                                        to.counted = True
                                        print('ID '+ str(to.objectID) + ' going up' + ' direction : ' + str(direction) + ' centroid : ' + str(centroid) + ' pixcel compared to : ' + str(i[0]) + ' ' + str(i[1]))
                                        if enter_direction == 'up':
                                            check_exceed(x,frame)
                                        break

                            
                            elif direction > 0:
                                for i in iterlist:
                                    if centroid[1] > i[1]:
                                        var.totalDown += 1
                                        var.empty1.append(var.totalDown)
                                        to.counted = True
                                        print('ID '+ str(to.objectID) + ' going down' + ' direction : ' + str(direction) + ' centroid : ' + str(centroid) + ' pixcel compared to : ' + str(i[0]) + ' ' + str(i[1]))
                                        if enter_direction == 'down':
                                            check_exceed(x,frame)
                                        break
                                        
                        elif 0 < m < 1000000000 and vertical_direction == 1:
                            
                            if direction < 0:
                                for i in iterlist:
                                    if centroid[0] < i[0] and centroid[1] < i[1]:
                                
                                        var.totalUp += 1
                                        var.empty.append(var.totalUp)
                                        to.counted = True
                                        print('ID '+ str(to.objectID) + ' going up' + ' direction : ' + str(direction) + ' centroid : ' + str(centroid) + ' pixcel compared to : ' + str(i[0]) + ' ' + str(i[1]))
                                        if enter_direction == 'up':
                                            check_exceed(x,frame)
                                        break

                            
                            elif direction > 0:
                                for i in iterlist:
                                    if centroid[0] > i[0] and centroid[1] > i[1]:
                                        var.totalDown += 1
                                        var.empty1.append(var.totalDown)
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
                                
                                        var.totalUp += 1
                                        var.empty.append(var.totalUp)
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
                                        var.totalDown += 1
                                        var.empty1.append(var.totalDown)
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
                                
                                        var.totalUp += 1
                                        var.empty.append(var.totalUp)
                                        to.counted = True
                                        print('ID '+ str(to.objectID) + ' going left' + ' direction : ' + str(direction) + ' centroid : ' + str(centroid) + ' pixcel compared to : ' + str(i[0]) + ' ' + str(i[1]))
                                        if enter_direction == 'left':
                                            check_exceed(x,frame)
                                        break

                            
                            elif direction > 0:
                                for i in iterlist:
                                    if centroid[0] > i[0]:
                                        var.totalDown += 1
                                        var.empty1.append(var.totalDown)
                                        to.counted = True
                                        print('ID '+ str(to.objectID) + ' going right' + ' direction : ' + str(direction) + ' centroid : ' + str(centroid) + ' pixcel compared to : ' + str(i[0]) + ' ' + str(i[1]))
                                        if enter_direction == 'right':
                                            check_exceed(x,frame)
                                        break
                                        
                        elif 0 < m < 1000000000 and vertical_direction == 0:
                            
                            if direction < 0:
                                for i in iterlist:
                                    if centroid[0] < i[0] and centroid[1] < i[1]:
                                
                                        var.totalUp += 1
                                        var.empty.append(var.totalUp)
                                        to.counted = True
                                        print('ID '+ str(to.objectID) + ' going left' + ' direction : ' + str(direction) + ' centroid : ' + str(centroid) + ' pixcel compared to : ' + str(i[0]) + ' ' + str(i[1]))
                                        if enter_direction == 'left':
                                            check_exceed(x,frame)
                                        break

                            
                            elif direction > 0:
                                for i in iterlist:
                                    if centroid[0] > i[0] and centroid[1] > i[1]:
                                        var.totalDown += 1
                                        var.empty1.append(var.totalDown)
                                        to.counted = True
                                        print('ID '+ str(to.objectID) + ' going right' + ' direction : ' + str(direction) + ' centroid : ' + str(centroid) + ' pixcel compared to : ' + str(i[0]) + ' ' + str(i[1]))
                                        if enter_direction == 'right':
                                            check_exceed(x,frame)
                                        break
                                        
                    x = []
                    # compute the sum of total people inside
                    if enter_direction == 'down' or enter_direction == 'right':
                        x.append(len(var.empty1)-len(var.empty))
                    else:
                        x.append(len(var.empty)-len(var.empty1))
                    #print("Total people inside:", x)




            # store the trackable object in our dictionary
            var.trackableObjects[objectID] = to

        if enter_direction == 'down' or enter_direction == 'right':
            info = [
            ("Exit", var.totalUp),
            ("Enter", var.totalDown),
            ]
        else:
            info = [
            ("Exit", var.totalDown),
            ("Enter", var.totalUp),        
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
                    try:
                        print("[INFO] Sending email alert..")
                        Mailer().send(config.MAIL)
                        print("[INFO] Alert sent")
                    except:
                        pass
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

        if config.Scheduler:
            if datetime.datetime.now() >= tmr :
                print('renew program')
                raise KeyboardInterrupt




cap.release()
cv2.destroyAllWindows()
#print('finished')
def start_thread():
    q= Queue()
    cam_process = Process(target=capture, args=(q,))
    cam_process.start()
    think_process = Process(target=tracker_peo, args=(q,))
    think_process.start()
    cam_process.join()
    think_process.join()

if config.Scheduler:
    ##Runs for every 1 second
    schedule.every(1).seconds.do(start_thread)
    global tmr
    
    ##Runs at every day (9:00 am). You can change it.
    #schedule.every().day.at("9:00").do(run)
    while 1:
        tmr=datetime.datetime.now()
        try:
            #tmr=tmr.replace(day=tmr.day + 1, hour=21, minute=12, second=0, microsecond=0)
            tmr=tmr.replace(day=tmr.day + 1, hour=0, minute=0, second=0, microsecond=0)
        except ValueError:
            try:
                tmr=tmr.replace(month=tmr.month + 1, day= 1,hour=0, minute=0, second=0, microsecond=0)
            except ValueError:
                tmr=tmr.replace(year= tmr.year + 1 ,month= 1, day= 1,hour=0, minute=0, second=0, microsecond=0)
        #print(tmr)
        #print(datetime.datetime.now())
        try:
            schedule.run_pending()

            if datetime.datetime.now() >= tmr:
                print('renew program')
                raise ValueError
                
        except:
            print('schedule error')
            continue
#else:
    #run()
