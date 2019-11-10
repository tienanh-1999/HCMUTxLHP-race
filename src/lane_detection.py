import cv2
import numpy as np
import matplotlib.pyplot as plt

def roi(img, vertices):
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, vertices, 255)
    masked = cv2.bitwise_and(img, mask)
    return masked
def draw_lines(img,lines):
    for line in lines:
        coords = line[0]
        cv2.line(img, (coords[0], coords[1]), (coords[2], coords[3]), [255,255,255], 3)
def draw_lane(img,lines):
    for line in lines:
        x,y = line[0],line[1]
        cv2.circle(img, (x,y), 0, (255,255,255), -1)
def draw_contours(img, contours):
    for contour in contours:
        for point in contour:
            coords = point[0]
            x,y = coords[0],coords[1]
            cv2.circle(img, (x,y), 0, (255,255,255), -1)
def image_process(img):
    kernel = np.ones((5,5),np.uint8)
    kernel2 = np.ones((1,1),np.uint8)
    blank_window = np.zeros((240,320),np.uint8)
    vertices = np.array([[0,240],[0,90],[320,90],[320,240],], np.int32)
    processed_img = roi(img, [vertices])
    edges = cv2.Canny(processed_img,240,250)
    closed_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    erosion = cv2.erode(closed_edges,kernel2,iterations = 1)
    ret, thresh = cv2.threshold(erosion, 127, 255, 0)
    contours,hie = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    final_contours = [x for x in contours if len(x)>50]
    draw_contours(blank_window, final_contours)
    
    dilation = cv2.dilate(blank_window,kernel,iterations = 1)
    blank_window = cv2.morphologyEx(dilation, cv2.MORPH_CLOSE, kernel)
    plt.imshow(blank_window, cmap='gray')
    plt.title('Test'), plt.xticks([]), plt.yticks([])
    plt.show()
    return blank_window
def lane_detect(img):
    left_lane, right_lane = [],[]
    for i in range(239,-1,-1):
        row = img[i]
        center = row[150:170]
        if (len(np.where(center == 255)[0])):
            break
        left_part,right_part = row[0:160],row[160:319]
        left_lane += [[np.where(left_part==255)[0][-1],i]] if (len(np.where(left_part==255)[0])>0)else [[0,i]]
        right_lane += [[np.where(right_part==255)[0][0]+160,i]] if (len(np.where(right_part==255)[0])>0)else [[320,i]]
    blank_window = np.zeros((240,320),np.uint8)
    draw_lane(blank_window,left_lane)
    draw_lane(blank_window,right_lane)
    plt.imshow(blank_window, cmap='gray')
    plt.title('Test'), plt.xticks([]), plt.yticks([])
    plt.show()
    return left_lane,right_lane
def poly_fit(line):
    x,y = np.array([]),np.array([])
    for point in line :
        x=np.append(x,[point[0]])
        y=np.append(y,[240-point[1]])
    poly =np.polyfit(y,x,2)
    return poly

if __name__ == "__main__":
    img = cv2.imread('img91.jpeg',cv2.IMREAD_GRAYSCALE)
    test = image_process(img)
    left_lane,right_lane = lane_detect(test)
    # print(poly_fit(left_lane),poly_fit(right_lane))  
    # x = np.linspace(0,240,1000)
    # y=((1 + (2*poly_fit(left_lane)[0]*x + poly_fit(left_lane)[1])**2)**1.5) / np.absolute(2*poly_fit(left_lane)[0])
    # #y = (x**2)*poly_fit(left_lane)[0]+poly_fit(left_lane)[1]*x+poly_fit(left_lane)[2]  
    # z = (x**2)*poly_fit(right_lane)[0]+poly_fit(right_lane)[1]*x+poly_fit(right_lane)[2]
    # plt.plot(x, y, 'r') # plotting t, a separately 
    # plt.plot(x, z, 'b') # plotting t, b separately 
    # plt.show()
