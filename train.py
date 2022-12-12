import tkinter as tk
import cv2,os
import csv
import numpy as np
from PIL import Image
import pandas as pd
import datetime
import time

def clearId():
    txtId.delete(0, 'end')    
    res = ""
    message.configure(text= res)

def clearName():
    txtName.delete(0, 'end')    
    res = ""
    message.configure(text= res)    
    
def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass
 
    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass
 
    return False
 
def TakeImages():        
    Id=(txtId.get())
    name=(txtName.get())
    if(is_number(Id) and name != ""):
        cam = cv2.VideoCapture(0)
        harcascadePath = "haarcascade_frontalface_default.xml"
        detector=cv2.CascadeClassifier(harcascadePath)
        sampleNum=0
        while(True):
            ret, img = cam.read()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(gray, 1.3, 5)
            for (x,y,w,h) in faces:
                cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)        
                #incrementing sample number 
                sampleNum=sampleNum+1
                #saving the captured face in the dataset folder TrainingImage
                cv2.imwrite("TrainingImage\ "+name +"."+Id +'.'+ str(sampleNum) + ".jpg", gray[y:y+h,x:x+w])
                #display the frame
                cv2.imshow('frame',img)
            #wait for 10 seconds 
            if cv2.waitKey(100) & 0xFF == ord('q'):
                break
            # break if the sample number is morethan 100
            elif sampleNum>100:
                break
        cam.release()
        cv2.destroyAllWindows() 
        res = "Images Saved for ID : " + Id +" Name : "+ name
        row = [Id , name]
        with open('StudentDetails\StudentDetails.csv','a+') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerow(row)
        csvFile.close()
        message.configure(text= res)
    else:
        if(is_number(Id) == False):
            res = "Enter Numeric Id"
            message.configure(text= res)
        if(name.isalpha() == False):
            res += " Enter Alphabetical Name"
            message.configure(text= res)
    
def TrainImages():
    recognizer = cv2.face_LBPHFaceRecognizer.create()
    faces,Id = getImagesAndLabels("TrainingImage")
    recognizer.train(faces, np.array(Id))
    recognizer.save("TrainingImageLabel\Trainner.yml")
    res = "Image Trained"
    message.configure(text= res)

def getImagesAndLabels(path):
    #get the path of all the files in the folder
    imagePaths=[os.path.join(path,f) for f in os.listdir(path)] 
    #print(imagePaths)
    
    #create empth face list
    faces=[]
    #create empty ID list
    Ids=[]
    #now looping through all the image paths and loading the Ids and the images
    for imagePath in imagePaths:
        #loading the image and converting it to gray scale
        pilImage=Image.open(imagePath).convert('L')
        #Now we are converting the PIL image into numpy array
        imageNp=np.array(pilImage,'uint8')
        #getting the Id from the image
        Id=int(os.path.split(imagePath)[-1].split(".")[1])
        # extract the face from the training image sample
        faces.append(imageNp)
        Ids.append(Id)        
    return faces,Ids

def TrackImages():
    recognizer = cv2.face.LBPHFaceRecognizer_create()#cv2.createLBPHFaceRecognizer()
    recognizer.read("TrainingImageLabel\Trainner.yml")
    harcascadePath = "haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(harcascadePath);    
    df=pd.read_csv("StudentDetails\StudentDetails.csv")
    cam = cv2.VideoCapture(0)
    font = cv2.FONT_HERSHEY_SIMPLEX        
    col_names =  ['Id','Name','Date','Time']
    attendance = pd.DataFrame(columns = col_names)    
    while True:
        ret, im =cam.read()
        gray=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
        faces=faceCascade.detectMultiScale(gray, 1.2,5)    
        for(x,y,w,h) in faces:
            cv2.rectangle(im,(x,y),(x+w,y+h),(225,0,0),2)
            Id, conf = recognizer.predict(gray[y:y+h,x:x+w])                                   
            if(conf < 60):
                ts = time.time()      
                date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
                timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
                aa=df.loc[df['Id'] == Id]['Name'].values
                tt=str(Id)+"-"+aa
                attendance.loc[len(attendance)] = [Id,aa,date,timeStamp]
                
            else:
                Id='Unknown'                
                tt=str(Id)  
            # if(conf > 50):
            #     noOfFile=len(os.listdir("ImagesUnknown"))+1
            #     cv2.imwrite("ImagesUnknown\Image"+str(noOfFile) + ".jpg", im[y:y+h,x:x+w])            
            cv2.putText(im,str(tt),(x,y+h), font, 1,(255,255,255),2)        
        attendance=attendance.drop_duplicates(subset=['Id'],keep='first')    
        cv2.imshow('im',im) 
        if (cv2.waitKey(1)==ord('q')):
            break
    ts = time.time()      
    date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
    timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
    Hour,Minute,Second=timeStamp.split(":")
    fileName="Attendance\Attendance_"+date+"_"+Hour+"-"+Minute+"-"+Second+".csv"
    attendance.to_csv(fileName,index=False)
    cam.release()
    cv2.destroyAllWindows()
    res=attendance
    messageAttendance.configure(text= res)


#create windown
window = tk.Tk()
window.title("Nhận diện khuôn mặt")
 
window.geometry('850x640')
window.configure(background='grey')

window.grid_rowconfigure(0, weight=1)
window.grid_columnconfigure(0, weight=1)

#display window

#Title display windown
message = tk.Label(window, text="Nhận diện khuôn mặt"  ,fg="black", bg="grey"  ,font=('times', 30, 'italic bold underline')) 
message.place(x=230, y=50)

#button take image
takeImg = tk.Button(window, text="Take Images", command=TakeImages, width=10, height=1, activebackground = "Red", font=('times', 15, ' bold '))
takeImg.place(x=50, y=200)

#button train image
trainImg = tk.Button(window, text="Train Images", command=TrainImages, width=10, height=1, activebackground = "Red", font=('times', 15, ' bold '))
trainImg.place(x=50, y=300)

#button track image
trackImg = tk.Button(window, text="Track Images", command=TrackImages, width=10, height=1, activebackground = "Red", font=('times', 15, ' bold '))
trackImg.place(x=50, y=400)

#button quit
quitWindow = tk.Button(window, text="Quit", command=window.destroy, width=10, height=1, activebackground = "Red", font=('times', 15, ' bold '))
quitWindow.place(x=50, y=500)

#title Id
lbId = tk.Label(window, text="Enter ID:",width=20  ,height=2, bg="grey"  ,font=('times', 15, ' bold ') ) 
lbId.place(x=300, y=195)

#input box Id
txtId = tk.Entry(window,width=20  ,bg="white" ,font=('times', 15, ' bold '))
txtId.place(x=500, y=208)

#button clear Id
clearButton = tk.Button(window, text="Clear", command=clearId, width=5, height=1, activebackground = "Red", font=('times', 15, ' bold '))
clearButton.place(x=720, y=200)

#title Name
lbl2 = tk.Label(window, text="Enter Name:",width=20  ,bg="grey"    ,height=2 ,font=('times', 15, ' bold ')) 
lbl2.place(x=300, y=295)

#input box Name
txtName = tk.Entry(window,width=20  ,bg="white"  ,font=('times', 15, ' bold ')  )
txtName.place(x=500, y=308)

#button clear Name
clearButton2 = tk.Button(window, text="Clear", command=clearName, width=5, height=1, activebackground = "Red", font=('times', 15, ' bold '))
clearButton2.place(x=720, y=300)

#title Notification
lbl3 = tk.Label(window, text="Notification: ",width=20  ,bg="grey"  ,height=2 ,font=('times', 15, ' bold ')) 
lbl3.place(x=300, y=400)

#output Notification
message = tk.Label(window, text="" ,bg="grey"  ,width=30  ,height=2, activebackground = "yellow" ,font=('times', 15, ' bold ')) 
message.place(x=500, y=400)

#title Attendance
lbl3 = tk.Label(window, text="Attendance : ",width=20  ,bg="grey"  ,height=2 ,font=('times', 15, ' bold  underline')) 
lbl3.place(x=300, y=500)

#output Attendance
messageAttendance = tk.Label(window, text="" ,bg="grey",activeforeground = "green",width=30  ,height=2  ,font=('times', 15, ' bold ')) 
messageAttendance.place(x=500, y=500)

window.mainloop()