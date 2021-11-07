import datetime
#===============================================================================
""" Optional features config. """
#===============================================================================
# Enter mail below to receive real-time email alerts
# e.g., 'email@gmail.com'
MAIL = ''
# Enter the ip camera url (e.g., url = 'http://191.138.0.100:8040/video')
url = 0
#rtsp://admin:Hdit23059797@192.168.13.125:554/ISAPI/Streaming/channels/101
# ON/OFF for mail feature. Enter True to turn on the email alert feature.
ALERT = False
# Set max. people inside limit. Optimise number below: 10, 50, 100, etc.
Threshold = 100
# Threading ON/OFF
Thread = False
############################
# Simple log to log the counting data
Log = True
# choose the time to updateLog below, for example if we want to update every 1 hour it would be write as timedel= datetime.timedelta(seconds=0,minutes=0,hours=1,days=0,weeks=0)
timedel= datetime.timedelta(seconds=0,minutes=0,hours=1,days=0,weeks=0)
# camera place name
cam_place='1st floor'
####################
# Auto run/Schedule the software to run at your desired time
Scheduler = False
# Auto stop the software after certain a time/hours
Timer = False
#requests config
five_mins= True
people_change= True
request_API='http://admin:hdit23059797@192.168.13.250:7001/api/createEvent/'
#===============================================================================
#===============================================================================
