
import numpy as np
import cv2

# img1 = cv2.imread('images/in000318.jpg')
# img2 = cv2.imread('images/in000419.jpg')

# img1 = cv2.imread('images/in000595.jpg')
# img2 = cv2.imread('images/in000658.jpg')

img1 = cv2.imread('images/IMG_0758_small.JPG')
img2 = cv2.imread('images/IMG_0761_small.JPG')


img1gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
img2gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)


# Non Maximum Suppression routine from: http://www.pyimagesearch.com/2015/02/16/faster-non-maximum-suppression-python/

# Malisiewicz et al.
def non_max_suppression_fast(boxes, overlapThresh):
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []

    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")

    # initialize the list of picked indexes
    pick = []

    # grab the coordinates of the bounding boxes
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]

    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)

    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)

        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])

        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)

        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]

        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last],
            np.where(overlap > overlapThresh)[0])))

    # return only the bounding boxes that were picked using the
    # integer data type
    return boxes[pick].astype("int")



# bilateral filtering
img1gray = cv2.bilateralFilter(img1gray,9,75,75)
img2gray = cv2.bilateralFilter(img2gray,9,75,75)

cv2.imwrite('img1_blur.png', img1gray)
cv2.imwrite('img2_blur.png', img2gray)

# image difference
img_diff_gray = cv2.absdiff(img1gray, img2gray)
cv2.imwrite('img_diff_gray.png', img_diff_gray)

# binary threshold on image difference
ret, img_diff_thresh = cv2.threshold(img_diff_gray, 40, 255,0)
cv2.imwrite('img_diff_thresh.png', img_diff_thresh)

# morphology open on image difference
kernel_mopen = np.ones((3,3),np.uint8)
img_diff_thresh = cv2.morphologyEx(img_diff_thresh, cv2.MORPH_OPEN, kernel_mopen)
cv2.imwrite('img_diff_thresh_morph1.png', img_diff_thresh)

# morphology close on image difference
kernel_mclose = np.ones((7,7),np.uint8)
img_diff_thresh = cv2.morphologyEx(img_diff_thresh, cv2.MORPH_CLOSE, kernel_mclose)
cv2.imwrite('img_diff_thresh_morph2.png', img_diff_thresh)

# find contours on image difference
contours,hierarchy = cv2.findContours(img_diff_thresh,3,1)
cv2.imwrite('img_contours.png', img_diff_thresh)

# find bounding rectangles for contours
img_h, img_w = img_diff_gray.shape
rect_list_orig = []

for i in xrange (0, len(contours)):
    x1,y1,w,h = cv2.boundingRect(contours[i])
    border = 5
    x2 = x1 + w
    y2 = y1 + h
    if x2 + border < img_w: x2 += border
    if y2 + border < img_h: y2 += border
    if x1 - border > 0: x1 -= border
    if y1 - border > 0: y1 -= border
    rect_list_orig.append((x1,y1,x2,y2))
    # cv2.rectangle(img_diff_gray,(x1,y1),(x2,y2),(255,255,255),1)


# print 'original rects found: ', rect_list_orig
rects_array_orig = np.array(rect_list_orig)
rects_list_nms = []

# perform non maximum suppression on bounding rectangles
rects_array_nms = non_max_suppression_fast(rects_array_orig, 0.2)

print "after applying non-maximum suppression, %d bounding boxes" % (len(rects_array_nms))

for (startX, startY, endX, endY) in rects_array_nms:
    # ignore small bounding rectangles
    if (endX - startX) * (endY - startY) > 300:
        rects_list_nms.append((startX, startY, endX, endY))
        cv2.rectangle(img_diff_gray, (startX, startY), (endX, endY), (255, 255, 255), 1)

cv2.imwrite('img_diff_gray_rect.png', img_diff_gray)


# - extract bounding rectangle area from each image
# - find number of features (SIFT) within the extracted area
# - store reference to image with the lowest number of SIFT features in the bounding rectangle
rects_dict_best = {}

for i1,image in enumerate([img1gray, img2gray]):
    for i2, rect in enumerate(rects_list_nms):
        x1,y1,x2,y2 = rect
        crop_img = image[y1: y2, x1: x2]
        print 'rect: ', rect
        sift = cv2.SIFT()
        kp = sift.detect(crop_img,None)
        kp_count = len(kp)
        print 'num keypoints: ', len(kp)

        if rect in rects_dict_best:
            if kp_count < rects_dict_best[rect][0]:
                rects_dict_best[rect] = (kp_count, i1)
        else:
            rects_dict_best[rect] = (kp_count, i1)

# print rects_dict_best

# replace bounding rectangle areas with those having the lowest number of SIFT features
image_list = [img1, img2]
for rect in rects_dict_best:
    if rects_dict_best[rect][1] != 0:
        x1, y1, x2, y2 = rect
        s_img = image_list[rects_dict_best[rect][1]][y1: y2, x1: x2]
        x_offset=x1
        y_offset=y1
        img1[y_offset:y_offset+s_img.shape[0], x_offset:x_offset+s_img.shape[1]] = s_img

cv2.imwrite('final.png', img1)


