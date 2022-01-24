import numpy as np
import cv2

class main:
    def __init__(self):
        self.x1, self.y1 = 0,0
        self.cap = cv2.VideoCapture('Pixals.mp4')
        self.logo=cv2.imread('opencv_logo.png')
        s = self.logo.shape
        self.kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
        # gl = cv2.cvtColor(logo, cv2.COLOR_BGR2GRAY)
        white = 255 * np.ones((s[0],s[1]),np.uint8)
        black = np.zeros((s[0],s[1]),np.uint8)

    def click_event(self, event, x,y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            global x1,y1
            self.x1,self.y1 = x,y
    
    def resize(self, img):
        return cv2.resize(img,(512,512))

    def foreground(self, cap="", logo="", out = "",x1=-1,y1=-1):
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
            cv2.setMouseCallback('image',self.click_event)
            cv2.waitKey(1000)
            x1,y1 = self.x1, self.y1

        s = logo.shape
        if y1+s[0] > frame1.shape[0] :
            y1 = (frame1.shape[0]-s[0])
        if x1+s[1] > frame1.shape[1] :
            x1 = (frame1.shape[1]-s[1])

        count = 0
        while cap.isOpened():
            count+=1

            diff = cv2.absdiff(frame1, back)
            diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

            diff = cv2.erode(diff, self.kernel,iterations=3)
            diff = cv2.dilate(diff, self.kernel,iterations=3)
            diff = cv2.inRange(diff, 20, 255)

            cont,_ = cv2.findContours(diff,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

            mask = diff[y1:y1+s[0],x1:x1+s[1]]
            frame1_ = frame1[y1:y1+s[0],x1:x1+s[1]]
            logo_ = np.where(logo==(0,0,0),frame1_, logo)

            frame1[y1:y1+s[0],x1:x1+s[1]] = cv2.bitwise_and(frame1_,frame1_, mask=mask) + cv2.bitwise_and(logo_, logo_, mask= cv2.bitwise_not(mask))
            frame2[y1:y1+s[0],x1:x1+s[1]] = logo_

            if ret==True:
                cv2.imshow("frame1", self.resize(frame1))
                cv2.imshow("frame2", self.resize(frame2))
                # result = np.hstack([resize(frame2), resize(frame1)])
                if out != "":
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

x = main()
x.foreground(logo = cv2.imread("wingzz.jpeg"),x1=500,y1=500)
# x.foreground(logo = 255*np.ones((1080,1920,3),np.uint8))