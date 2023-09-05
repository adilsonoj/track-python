import cv2
import numpy as np

camera = cv2.VideoCapture("carros3.mp4")

TEXT_COLOR = (0, 255, 0)
TRACKER_COLOR = (255, 0, 0)
FONT = cv2.FONT_HERSHEY_COMPLEX
minArea = 3000



fgbgKnn = cv2.createBackgroundSubtractorKNN()


def getCenter(x, y, w, h):
    x1 = int(w / 2)
    y1 = int(h / 2)
    cx = x + x1
    cy = y + y1
    return cx,cy

def getFilter(frame, filter):
    if filter == 'closing':
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
        return cv2.morphologyEx(frame, cv2.MORPH_CLOSE, kernel, iterations=2)
    
    if filter == 'opening':
        # kernel = np.ones((5,5),np.uint8)
        return cv2.morphologyEx(frame, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5)), iterations=2)
    
    if filter == 'dilation':
        kernel = np.ones((5,5),np.uint8)
        return cv2.morphologyEx(frame, kernel, iterations=2)
    
    if filter == 'combine':
        kernel = np.ones((2,2),np.uint8)

        opening = cv2.morphologyEx(frame, cv2.MORPH_OPEN, kernel, iterations=4)
        closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel, iterations=2)
        dilation = cv2.morphologyEx(closing, cv2.MORPH_DILATE, kernel, iterations=5)

        return dilation

def main():
    fgbg = cv2.createBackgroundSubtractorMOG2()
    detects = []
    up=0
    down=0
    total=0
    while camera.isOpened:
        check, frame = camera.read()
        if not check:
            print("Erro")
            break

        frame = cv2.resize(frame, (0,0), fx=0.5, fy=0.5)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        fmask = fgbg.apply(gray)


        retval, th = cv2.threshold(fmask, 100, 255, cv2.THRESH_BINARY)

        fmask = getFilter(th, 'combine')

        # fgbg = cv2.medianBlur(fgbg, 5)
        # result = cv2.bitwise_and(frame, frame, mask=fgbg)

        # adicionar linha
        dimension = frame.shape
        height = dimension[0]
        width = dimension[1]

        middle = int(height/2)

        cv2.line(frame,(0,middle), (width, middle), TEXT_COLOR, 2 )

        contours, hierarchy = cv2.findContours(fmask, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

        count = 0
        for cnt in contours:
            area = cv2.contourArea(cnt)
            (x,y,w,h) = cv2.boundingRect(cnt)

            # verificar a area é suficiente, eliminar ruido
            if(area > minArea):
                centro = getCenter(x,y,w,h)

                # 4=tamanho | -1 preenchido
                cv2.circle(frame, centro, 4, (0,0,255), -1)
                cv2.rectangle(frame,(x,y), (x+w,y+h),TRACKER_COLOR, 2)
                cv2.putText(frame, str(count), (x-5, y-5), FONT, 0.5, (0,255,255),2)


                if len(detects) <= 1:
                    detects.append([])
                
                # if centro[1] > middle-30 and centro[1] < middle+30:
                detects[count].append(centro)

                print(count)
                
                count += 1


        # if len(contours) ==0:
        #     detects.clear()
        else:
            for dt in detects:
                for (i,el) in enumerate(dt):
                    if dt[i-1][1] < middle and el[1] > middle:
                        dt.clear()
                        up+=1
                        total+=1
                        print('subiu')
                        continue

                    if dt[i-1][1] > middle and el[1] < middle:
                        dt.clear()
                        down+=1
                        total+=1
                        print('desceu')
                        continue
           
                       
                

        cv2.imshow('Frame', frame)
        # cv2.imshow('gray', gray)
        cv2.imshow('fmask', fmask)
        # cv2.imshow('th', th)
   


        if cv2.waitKey(50) & 0xFF == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()


main()