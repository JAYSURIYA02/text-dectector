import torch
import torchvision
print(torch.__version__)
print(torchvision.__version__)

import cv2
import easyocr
import matplotlib.pyplot as plt


reader = easyocr.Reader(['en'], gpu=True)
print("1.image\n2.video\n3.webcam")

choice = input ('enter the choice ').strip()
if(int(choice)==1):
# Load with OpenCV
    path = input('enter the image path ').strip() 
    img_bgr = cv2.imread(path)
    
    # OCR
    result = reader.readtext(img_bgr)
    threshold = 0.25
    for (bbox, text, score) in result:
        print(bbox, text, score)
        if score >threshold:
            cv2.rectangle(img_bgr,bbox[0],bbox[2],(0,255,0),5)
            cv2.putText(img_bgr,text,bbox[0],cv2.FONT_HERSHEY_COMPLEX,0.50,(0,0,255),2)
    plt.imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()
    
elif(int(choice) ==2 ):
    path = input('Enter the video path ').strip()
    frames = cv2.VideoCapture(path)
    while True:
        ret,img_bgr = frames.read()
        if not ret:
            exit()
        result = reader.readtext(img_bgr)
        threshold = 0.25
        for (bbox, text, score) in result:
            print(bbox, text, score)
            if score >threshold:
                pt1 = tuple(map(int, bbox[0]))  
                pt2 = tuple(map(int, bbox[2]))
                cv2.rectangle(img_bgr, pt1, pt2, (0, 255, 0), 2)
                cv2.putText(img_bgr, text, pt1, cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 2)
        cv2.imshow("Video OCR", img_bgr)

    # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    frames.release()
    cv2.destroyAllWindows()

elif(int(choice) == 3):
    cam = cv2.VideoCapture(0)
    while True:
        ret,img_bgr = cam.read()
        if not ret:
            exit()
        result = reader.readtext(img_bgr)
        threshold = 0.25
        for (bbox, text, score) in result:
            print(bbox, text, score)
            if score >threshold:
                pt1 = tuple(map(int, bbox[0]))  
                pt2 = tuple(map(int, bbox[2]))
                cv2.rectangle(img_bgr, pt1, pt2, (0, 255, 0), 2)
                cv2.putText(img_bgr, text, pt1, cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 2)
        cv2.imshow("Webcam OCR", img_bgr)
    # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cam.release()
    cv2.destroyAllWindows()
else:
    print("invalid choice")