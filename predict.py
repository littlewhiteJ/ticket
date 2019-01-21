import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import math
import instant_test as it
import scipy.ndimage as nd
import shutil

def takeSecond(elem):
    return elem[1]

def get_pred_boxes(sl, threshold):
    ff_map, _ = nd.measurements.label(sl)
    pred_boxes_tmp = nd.measurements.find_objects(ff_map)
    pred_boxes = []
    for ob in pred_boxes_tmp:
        size = (ob[0].stop-ob[0].start)*(ob[1].stop-ob[1].start)
        if size > threshold:
            h = ob[0].stop-ob[0].start
            w = ob[1].stop-ob[1].start
            pred_boxes.append([ob[0].start,ob[1].start,ob[0].stop-ob[0].start,ob[1].stop-ob[1].start])
    return pred_boxes


#####################################
dirpath = 'test_data/'  
#####################################


if os.path.exists('cut_pic/'):
    shutil.rmtree('cut_pic/')
os.mkdir('cut_pic/')


imgfiles = os.listdir(dirpath)

print('start cut and rotate')

for imgfile in imgfiles:
    # print(imgfile)
    fullname = dirpath + imgfile
    img = cv2.imread(fullname)
    
    gray_img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    blur_img = cv2.medianBlur(gray_img,7)
    #cv2.imwrite('blur.png', blur_img)
    
    _ ,thres_img = cv2.threshold(blur_img,100,255,cv2.THRESH_BINARY)
    #cv2.imwrite('threshold.png', thres_img)
    
    thres_kernel = np.ones((30,30), np.uint8)
    closing_img = cv2.morphologyEx(thres_img,cv2.MORPH_CLOSE,thres_kernel)

    # to find the contours
    _,contours,hierarchy = cv2.findContours(closing_img,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    
    areas = []
    for contour in contours:
        areas.append(cv2.contourArea(contour))
    

    indexax = np.argmax(areas)
    '''
    print(indexax)
    print(areas)
    '''
    
    for i in range(len(areas)):
        if i != indexax:
            cv2.drawContours(closing_img, contours, i, (0,0,0), thickness=-1)

    edges_img = cv2.Canny(closing_img,300,400)

    edge_kernel = np.ones((5,5), np.uint8)
    edges_img = cv2.dilate(edges_img, edge_kernel, iterations=1)
    # cv2.imwrite('dilation_edges.png', edges_img)

    lines = cv2.HoughLinesP(edges_img, rho = 1, theta = np.pi/180, threshold = 100, minLineLength = 500, maxLineGap = 200)
    lines = np.squeeze(lines).tolist()

    angle = 0
    for line in lines:
        x1 = line[0]
        y1 = line[1]
        x2 = line[2]
        y2 = line[3]

        if x1 == x2:
            continue
        atan_angle = 180 / np.pi * math.atan((y2 - y1) / (x2 - x1))
        if abs(atan_angle) > 20:
            continue
        else:    
            angle = atan_angle + 90
            break

    # print(angle)


    (h,w) = img.shape[:2]
    center = (w / 2,h / 2)

    M = cv2.getRotationMatrix2D(center,angle,1)
    img_rotate = cv2.warpAffine(img,M,(w,h))
    mask_rotate = cv2.warpAffine(closing_img,M,(w,h))
    
    # print(img_rotate.shape)
    
    # g_mask = cv2.cvtColor(mask_rotate,cv2.COLOR_BGR2GRAY) 
    
    _, cs, _ = cv2.findContours(mask_rotate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
 
    for c in cs:
        x, y, w, h = cv2.boundingRect(c)
        img_out = img_rotate[y:y+h, x:x+w]
        # cv2.rectangle(img_rotate, (x, y), (x + w, y + h), (0, 255, 0), 2)


    '''
    savename = 'img_rotate.png'
    cv2.imwrite(savename, img_rotate)

    savename = 'mask_rotate.png'
    cv2.imwrite(savename, mask_rotate)
    '''
    
    _ ,thres = cv2.threshold(img_out,220,255,cv2.THRESH_BINARY)
    savename = 'thres/thres_' + imgfile
    cv2.imwrite(savename, thres)
    
    (hout, wout) = img_out.shape[:2]
    ks = 80
    img_sample1 = img_out[0:ks,0:wout]
    img_sample2 = img_out[hout - ks - 1:hout - 1, 0:wout]
    is1 = cv2.cvtColor(img_sample1,cv2.COLOR_BGR2GRAY)
    is2 = cv2.cvtColor(img_sample2,cv2.COLOR_BGR2GRAY)
    light1 = 0
    light2 = 0
    for i in is1:
        for j in i:
            light1 += j
    
    for i in is2:
        for j in i:
            light2 += j


    if light1 < light2:
        (h,w) = img_out.shape[:2]
        center = (w / 2,h / 2)

        M = cv2.getRotationMatrix2D(center,180,1)
        img_out = cv2.warpAffine(img_out,M,(w,h))


    savename = 'cut_pic/' + imgfile
    # print(savename)
    cv2.imwrite(savename, img_out)
    # break

print('cut and rotate done!')
print('start slice and predict')


#######################
stdl = 64
totallist = []
model = it.load_model()

#######################

root_dir = './cut_pic'
# hard code two regions in (h, w)
upper_lt = (20, 50)
upper_size = (80, 280)

lower_lt = (550, 40)
lower_size = (90, 470)

# hard code a threshold
size_threshold = 50*100
number_size_threshold = 10*5

file_list = os.listdir(root_dir)
for file_name in file_list:

    ori_image = cv2.imread(os.path.join(root_dir, file_name))
    image = ori_image[:, :, 0] # to grey image

    # heuristically get number region
    upper_region = image[upper_lt[0]:upper_lt[0]+upper_size[0], upper_lt[1]:upper_lt[1]+upper_size[1]]
    lower_region = image[lower_lt[0]:lower_lt[0]+lower_size[0], lower_lt[1]:lower_lt[1]+lower_size[1]]
    

    '''
    cv2.imshow('upper', upper_region)
    cv2.imshow('lower', lower_region)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    '''
    
    # find precise region
    greyscale_threshold = [150, 50]
    region_list = [upper_region, lower_region]
    iterations = [15, 7]
    new_regions = []
    for idx, sub_img in enumerate(region_list):
        sl = sub_img < greyscale_threshold[idx]
        iters = iterations[idx]
        sl = nd.morphology.binary_dilation(sl,iterations=iters)
        sl = nd.morphology.binary_fill_holes(sl)
        sl = nd.morphology.binary_erosion(sl,iterations=iters//3)
        pred_boxes = get_pred_boxes(sl, size_threshold)
        
        # first check whether there are more than one boxes per region
        if len(pred_boxes) > 1:
            pred_boxes = sorted(pred_boxes, key=lambda x: -x[2]*x[3]) # sort by size
            pred_boxes = pred_boxes[:2] # take first two boxes
            final_box = pred_boxes[0] if pred_boxes[0][2] > pred_boxes[1][2] else pred_boxes[1] # take the one with larger height
        else:
            final_box = pred_boxes[0]
        y, x, h, w = final_box
        # visualization code
        '''
        vis_img = np.expand_dims(sub_img, axis=2)
        vis_img = np.repeat(vis_img, 3, axis=2)
        print(final_box)
        cv2.rectangle(vis_img, (x, y), (x+w, y+h), (0, 255, 0), 1)
        cv2.imshow('tmp', vis_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        '''
        new_regions.append(sub_img[y:y+h, x:x+w])


    # segment each numbers
    region_list = new_regions
    greyscale_threshold = [120, 20]  # a new threshold
    nr_numbers = [7, 21]  # the number of letters
    max_iters = 5

    ##############################
    itemlist = []
    itemlist.append(file_name)
    ##############################
    
    for idx, sub_img in enumerate(region_list):
        flag = False
        for this_grey_iter in range(max_iters):  # try different grey threshold
            for this_nd_iter in range(max_iters):  # try different dilation/erosion iterations
                sl = sub_img < greyscale_threshold[idx] + 10*this_grey_iter 
                if this_nd_iter > 0:
                    sl = nd.morphology.binary_dilation(sl,iterations=this_nd_iter)
                    sl = nd.morphology.binary_fill_holes(sl)
                    sl = nd.morphology.binary_erosion(sl,iterations=this_nd_iter)

                pred_boxes = get_pred_boxes(sl, number_size_threshold)
                if len(pred_boxes) == nr_numbers[idx]: # hit
                    flag = True
                    break
            if flag: # hit
                break
            
        # visualization code
        # print(len(pred_boxes))
        vis_img = np.expand_dims(sub_img, axis=2)
        vis_img = np.repeat(vis_img, 3, axis=2)
        # print(vis_img.shape)
        gray_vis = cv2.cvtColor(vis_img,cv2.COLOR_BGR2GRAY)
        if idx == 0:
            continue

        if len(pred_boxes) != 21:
            continue 

        pred_boxes.sort(key=takeSecond)

        outstr = ''

        for i in range(21):
            y, x, h, w = pred_boxes[i]
            img_out = gray_vis[y:y+h, x:x+w]
            img_out = cv2.resize(img_out, (stdl, stdl))
            pre = it.instant_t(img_out, model)
            outstr += pre
            #cv2.rectangle(ori_image, (x_img, y_img), (x_img+w, y_img+h), (0, 255, 0), 1)
            

            '''
            print(y)
            img_out = gray_vis[y:y+h, x:x+w]
            savename = 'train_tag/' + str(save_tag) + '.png'
            cv2.imwrite(savename, img_out)
            save_tag += 1
            break
            '''
    
    itemlist.append(outstr)
    itemlist.append(outstr[14:21])   
    totallist.append(itemlist)

print('slice and predict done!')    
print('start write to file')

with open('prediction.txt', 'w') as f:
    for item in totallist:
        for i in item:
            f.write(i)
            f.write(' ')
        f.write('\n')

print('write to file done!')