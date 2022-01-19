import numpy as np
import cv2

global x1,y1
x1,y1 =900,500
cap=cv2.VideoCapture('Pixals.mp4')
logo=cv2.imread('opencv_logo.png')
logo = cv2.resize(logo, (256,256))
s = logo.shape

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))

gl = cv2.cvtColor(logo, cv2.COLOR_BGR2GRAY)
white = 255 * np.ones((s[0],s[1]),np.uint8)
black = np.zeros((s[0],s[1]),np.uint8)
mask_logo = np.array(np.where(gl==0,black,white))

def click_event(event, x,y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        #frame1[y:y+s[0],x:x+s[1]] = cv2.bitwise_or(frame1[y:y+s[0],x:x+s[1]],logo,mask=white)
        #cv2.imshow('image',frame1)
        #cv2.waitKey(1)
        global x1,y1
        x1,y1 = x,y
        print(x1,y1)
        
def resize(img):
    return cv2.resize(img,(512,512))

# frame = cv2.imread("frame.png")
ret, frame1 = cap.read()
back = frame1.copy()
frame2 = frame1.copy()

out = cv2.VideoWriter(
    'output.mp4',
    cv2.VideoWriter_fourcc(*'MJPG'),
    30.,
    (1024,512))

cv2.imshow('image', frame1)
cv2.setMouseCallback('image',click_event)
cv2.waitKey(1000)

count = 0
while cap.isOpened():
    count+=1
    
    hull = []
    #ret, frame2 = cap.read()
    diff = cv2.absdiff(frame1, back)
    diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    #diff = cv2.inRange(diff, 20, 255)

    #cv2.imshow("before morph", resize(diff))
    #diff = cv2.morphologyEx(diff, cv2.MORPH_CLOSE,kernel)
    diff = cv2.erode(diff, kernel,iterations=3)
    diff = cv2.dilate(diff, kernel,iterations=3)
    diff = cv2.inRange(diff, 20, 255)

    cont,_ = cv2.findContours(diff,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    #contSorted = sorted(cont, key=lambda x: cv2.contourArea(x))
    #cv2.drawContours(frame1, contSorted[-1], -1, (0,255,0),3)

    mask = diff[y1:y1+s[0],x1:x1+s[1]]
    frame1_ = frame1[y1:y1+s[0],x1:x1+s[1]]
    logo_ = np.where(logo==(0,0,0),frame1_, logo)
    #frame1[y1:y1+s[0],x1:x1+s[1]] = cv2.bitwise_and(frame1[y1:y1+s[0],x1:x1+s[1]],logo_ ,mask = mask)
    frame1[y1:y1+s[0],x1:x1+s[1]] = cv2.bitwise_and(frame1_,frame1_, mask=mask) + cv2.bitwise_and(logo_, logo_, mask= cv2.bitwise_not(mask))
    frame2[y1:y1+s[0],x1:x1+s[1]] = logo_

    if ret==True:
        #cv2.imshow('diff',resize(diff))
        cv2.imshow("frame1", resize(frame1))
        cv2.imshow("frame2",resize(frame2))
        result = np.hstack([resize(frame2), resize(frame1)])
        out.write(result.astype('uint8'))
    #frame1 = frame2

    print(len(cont), end=" ")
    ret,frame1=cap.read()
    if type(frame1) == np.ndarray:
        frame2 = frame1.copy()
    else:
        break

    if cv2.waitKey(1)==ord('q'):
        break
cv2.waitKey(0)
cv2.destroyAllWindows()
cap.release()
out.release()
# '''
