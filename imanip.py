import cv2



#https://pyimagesearch.com/2021/01/20/opencv-rotate-image/
def rotate_image(image, angle):
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center = (cX, cY), angle = angle, scale = 1.0) # angle in degree
    rotated = cv2.warpAffine(image, M, (w, h))
    image = rotated
    return image

#https://www.tutorialkart.com/opencv/python/opencv-python-resize-image/#gsc.tab=0
def resize_scale(image, scale): #100% -> 1:1
    width = int(image.shape[1] * scale / 100)
    height = int(image.shape[0] * scale / 100)
    dim = (width, height)
    
    image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
    return image

#https://www.tutorialkart.com/opencv/python/opencv-python-resize-image/#gsc.tab=0
def resize_dim(image, dim):
    image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
    return image

#https://stackoverflow.com/questions/41508458/python-opencv-overlay-an-image-with-transparency
def draw_overlay(image, x, y, top_image):
    x = int(x)
    y = int(y)
    alpha = top_image[:, :, 3] / 255.0
    
    image[y:y+top_image.shape[0], x:x+top_image.shape[1], 0] = (1. - alpha) * image[y:y+top_image.shape[0], x:x+top_image.shape[1], 0] + alpha * top_image[:, :, 0]
    image[y:y+top_image.shape[0], x:x+top_image.shape[1], 1] = (1. - alpha) * image[y:y+top_image.shape[0], x:x+top_image.shape[1], 1] + alpha * top_image[:, :, 1]
    image[y:y+top_image.shape[0], x:x+top_image.shape[1], 2] = (1. - alpha) * image[y:y+top_image.shape[0], x:x+top_image.shape[1], 2] + alpha * top_image[:, :, 2]
    return image