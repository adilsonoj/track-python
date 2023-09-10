import cv2

camera = cv2.VideoCapture("carros.mp4")

TEXT_COLOR = (0, 255, 0)
TRACKER_COLOR = (255, 0, 0)
FONT = cv2.FONT_HERSHEY_COMPLEX
min_area = 5000
max_area = 9000

def getCenter(x, y, w, h):
    x1 = int(w / 2)
    y1 = int(h / 2)
    cx = x + x1
    cy = y + y1
    return cx,cy

def zoom(frame):
    height, width, channels = frame.shape
    scale=22
    #prepare the crop
    centerX,centerY=int(height/2),int(width/2)
    radiusX,radiusY= int(scale*height/100),int(scale*width/100)

    minX,maxX=centerX-radiusX,centerX+radiusX
    minY,maxY=centerY-radiusY,centerY+radiusY

    cropped = frame[minX:maxX, minY:maxY]
    return cv2.resize(cropped, (0,0), fx=0.6, fy=0.6) 

def applyFilter(frame):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5)) 
        
        morph = cv2.morphologyEx(frame, cv2.MORPH_OPEN, kernel, iterations=2)
        morph = cv2.morphologyEx(morph, cv2.MORPH_CLOSE, kernel, iterations=6)
        morph = cv2.morphologyEx(morph, cv2.MORPH_DILATE, kernel, iterations=6)
        return morph

def main():
    fgbg = cv2.createBackgroundSubtractorMOG2()
    detects = []
    total=0
    offset = 7
    while camera.isOpened:
        check, frame = camera.read()
        
        frame = zoom(frame)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        blur = cv2.GaussianBlur(gray,(3,3),5)

        fmask = fgbg.apply(blur)

        retval, th = cv2.threshold(fmask, 110, 255, cv2.THRESH_BINARY)

        fmask = applyFilter(th)

        # adicionar linha
        dimension = frame.shape
        height = dimension[0]
        width = dimension[1]

        middle = int(height/2)

        cv2.line(frame,(0,middle), (width, middle), TEXT_COLOR, 2 )
        cv2.line(frame,(0,middle+30), (width, middle+30), TEXT_COLOR, 2 )
        cv2.line(frame,(0,middle-30), (width, middle-30), TEXT_COLOR, 2 )
        
 
        cv2.putText(frame, "VEICULOS: "+str(total), (10, 70), FONT, 0.5, (0,255,255),1)

        contours, hierarchy = cv2.findContours(fmask, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

        for (i, cnt) in enumerate(contours):
            area = cv2.contourArea(cnt)
            (x,y,w,h) = cv2.boundingRect(cnt)
            
            # verificar a area é suficiente, eliminar ruido
            if(area > min_area and area < max_area):
                centro = getCenter(x,y,w,h)

                # 4=tamanho | -1 preenchido
                cv2.circle(frame, centro, 4, (0,0,255), -1)
                cv2.rectangle(frame,(x,y), (x+w,y+h),TRACKER_COLOR, 2)

                if centro[1] < abs(middle+30) and centro[1] > abs(middle-30):
                    detects.append(centro)

                for (x,y) in detects:
                    if y <= abs(middle+offset) and y >= abs(middle-offset):
                        total+=1
                        cv2.line(frame,(0,middle), (width, middle), (0,255,255), 2 )
                        detects.remove((x,y))
                        print("Carros detectados:" +str(total))


        if len(contours) ==0:
            detects.clear()

        cv2.imshow('Frame', frame)
        # cv2.imshow('gray', gray)
        # cv2.imshow('blur', blur)
        cv2.imshow('fmask', fmask)
        # cv2.imshow('th', th)

        if cv2.waitKey(30) & 0xFF == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()


main()