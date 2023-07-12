import cv2 as cv
import backend

vid_i = cv.VideoCapture(0)
vid_i.set(3, 1280)
vid_i.set(4, 720)

# v_4cc = cv.VideoWriter_fourcc(*'AVC1')
# vid_o = cv.VideoWriter('./output.mp4', v_4cc, 25, (1280, 720))

while (read_image := vid_i.read())[0]:

    outimg = backend.process(read_image[1])
    window1 = "HAND TRACKING"
    cv.namedWindow(window1)
    
    cv.moveWindow(window1, 100, 50)
    cv.imshow(window1, outimg)
    # vid_o.write(outimg)
    key = chr(cv.waitKey(1) % 256)
    
    if key in 'qQ':
        break


vid_i.release()
# vid_o.release()
cv.destroyAllWindows()
