import numpy as np
import cv2

class Forex:
    # Change the background of your room wall
    # Add a funny sticker in the background
    # 
    # How to use
    # The first frame need to be the background.
    # So, everything that enters in the frame after that is captured.
    # The background can be modified in any way you want.
    # For modifying the background, change the kernel or apply more
    # filters as per need.

    def __init__(self):
        self.x1, self.y1 = 0,0 # default location of logo
        self.cap = cv2.VideoCapture('Pixals.mp4') # default video
        self.logo=cv2.imread('opencv_logo.png') # default logo
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7)) # default kernel(elliptical)

    def click_event(self, event, x,y, flags, param):
        # Choose the location of logo by left clicking at it
        if event == cv2.EVENT_LBUTTONDOWN:
            global x1,y1
            self.x1,self.y1 = x,y
    
    def resize(self, img):
        # Resize images or frames to a default size that fits best on the screen
        return cv2.resize(img,(512,512))

    def foreground(self, cap="", logo="", out = "",x1=-1,y1=-1):
        # cap   -> Video to be added
        # logo  -> logo or background that you want to add
        # out   -> Add the name of the saved video file (without extension)
        # x1,y1 -> Location of logo. If not stated, click_event function will be invoked

        if cap == "":
            cap = self.cap
        if logo == "":
            logo = self.logo
        ret, frame1 = cap.read()
        back = frame1.copy()
        frame2 = frame1.copy()

        if out != "":
            out = cv2.VideoWriter(
                f'{out}.mp4',
                cv2.VideoWriter_fourcc(*'MJPG'),
                30.,
                (1024,512))

        cv2.imshow('image', frame1)
        if x1==-1 or y1==-1:
            # If location is not explicitly defined, invoke click_event
            cv2.setMouseCallback('image',self.click_event)
            cv2.waitKey(1000)
            x1,y1 = self.x1, self.y1

        s = logo.shape
        # Adjust the logo if it doesn't fit on the screen
        if y1+s[0] > frame1.shape[0] :
            y1 = (frame1.shape[0]-s[0])
        if x1+s[1] > frame1.shape[1] :
            x1 = (frame1.shape[1]-s[1])

        count = 0
        while cap.isOpened():
            count+=1
            # Calculate the difference between first frame and the last frame to track the foreground
            diff = cv2.absdiff(frame1, back)
            diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

            # Apply morphological transformation to reduce noise
            diff = cv2.erode(diff, self.kernel,iterations=3)
            diff = cv2.dilate(diff, self.kernel,iterations=3)
            diff = cv2.inRange(diff, 20, 255)

            # cont,_ = cv2.findContours(diff,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

            mask = diff[y1:y1+s[0],x1:x1+s[1]]
            frame1_ = frame1[y1:y1+s[0],x1:x1+s[1]]
            
            # A logo is a rectangular shape while the actual logo might not
            # Replace the unwanted part in the logo image with the actual frame
            logo_ = np.where(logo==(0,0,0),frame1_, logo)

            frame1[y1:y1+s[0],x1:x1+s[1]] = cv2.bitwise_and(frame1_,frame1_, mask=mask) + cv2.bitwise_and(logo_, logo_, mask= cv2.bitwise_not(mask))
            frame2[y1:y1+s[0],x1:x1+s[1]] = logo_

            if ret==True:
                cv2.imshow("frame1", self.resize(frame1))
                cv2.imshow("frame2", self.resize(frame2))
                if out != "":
                    result = np.hstack([resize(frame2), resize(frame1)])
                    out.write(result.astype('uint8'))

            ret,frame1 = cap.read()
            if type(frame1) == np.ndarray:
                frame2 = frame1.copy()
            else:
                break

            if cv2.waitKey(1)==ord('q'):
                break
        cv2.waitKey(1)
        cv2.destroyAllWindows()
        cap.release()
        if out != "":
            out.release()

# Example cases
x = Forex()
x.foreground(logo = cv2.imread("wingzz.jpeg"),x1=500,y1=500)
# x.foreground(logo = 255*np.ones((1080,1920,3),np.uint8))