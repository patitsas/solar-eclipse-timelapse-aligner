import argparse
import errno
import json
import os
import shutil
import psutil
import sys
import time
from multiprocessing import Manager, Process, Queue, Value, freeze_support
from pathlib import Path

import cv2
import click
import numpy as np
from scipy import signal, spatial, stats
from tqdm import tqdm

import math
import doctest

#### constants
# colours
RED = (0, 0, 255)
ORANGE = (0, 128, 255)
GREEN = (0, 255, 0)





def get_sun_radius(sun_path, sun_threshold):
    '''(str) -> int
    Given filename path, return radius of sun using Han Lin's original method.
    >>> get_sun_radius('example0/ring.jpg', 25)
    127
    >>> get_sun_radius('example1/ex1.jpg', 25)
    110
    >>> get_sun_radius('example1/ex3.jpg', 55)
    110
    >>> get_sun_radius('example2/DSC05686.jpg', 60)
    141
    >>> get_sun_radius('example2/DSC05686.jpg', 25)
    0
    '''
    sun = cv2.imread(sun_path, cv2.IMREAD_COLOR)
    sun_gray = cv2.cvtColor(sun, cv2.COLOR_BGR2GRAY)
    _, sun_binary = cv2.threshold(sun_gray, sun_threshold, 255, cv2.THRESH_BINARY)
    _, _, stats, _ = cv2.connectedComponentsWithStats(sun_binary)
    sun_mask_r = int(max(stats[1][2:4]) // 2)
    
    return sun_mask_r


def get_centre_of_sun(fname, sun_threshold, sun_radius):
    '''str, dict, CV2 image -> int, int
    Find the centre of the sun in the image img.
    Return: x and y of the centre
    >>> get_centre_of_sun('example0/ring.jpg', 25, 127)
    (240, 180)
    >>> get_centre_of_sun('example1/ex1.jpg', 25, 110)
    (238, 153)
    >>> get_centre_of_sun('example1/ex3.jpg', 55, 110)
    (276, 132)
    >>> get_centre_of_sun('example2/DSC05686.jpg', 60, 141)
    (713, 1097)
    >>> get_centre_of_sun('example2/DSC05686.jpg', 25, 0)    
    Error: invalid radius for sun. Sun radius must be greater than 50 pixels.
    '''
    img = cv2.imread(fname, cv2.IMREAD_COLOR)
    
    # STAGE 1: FIND CENTRE OF SUN
    # grayscale
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # otsu
    # _, img_binary = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, img_binary = cv2.threshold(img_gray, sun_threshold, 255, cv2.THRESH_BINARY)
    # save sun_binary
    cv2.imwrite(fname.replace('/', '-sun-binary/'), img_binary)
    #print(np.sum(img_binary))

    sun_mask_r = sun_radius # 141 for my photos, 110 for OG's # sun radius
    sun_mask = np.zeros((sun_mask_r * 2 + 1, sun_mask_r * 2 + 1), np.float32) # making zeros on a canvas. sum: 0
    sun_mask = cv2.circle(sun_mask, (sun_mask_r, sun_mask_r), sun_mask_r, 1.0, -1) # this runs: img, centre, radius, colour, thickness
    #print(sun_mask)
    try:
        sun_mask = cv2.circle(sun_mask, (sun_mask_r, sun_mask_r), sun_mask_r - 50, 0.0, -1) # THIS is the source of the assertion error

        # fft convolve
        sun_signal = signal.fftconvolve(img_binary.astype('float') / 255, sun_mask, mode='same')
        # find coordinates of the largest value in sun_signal
        y, x = np.unravel_index(np.argmax(sun_signal, axis=None), sun_signal.shape)
        sun_x, sun_y = int(x), int(y) # these are the centre of the sun's circle
        
        return sun_x, sun_y
    except:
        if sun_radius < 50:
            print('Error: invalid radius for sun. Sun radius must be greater than 50 pixels.')


def output_encircled(img, path, sun_radius, sun_x, sun_y, original_moon_x = -1, original_moon_y = -1, moon_radius = -1, dirname = 'circled'):
    '''
    (CV2, str, int, int, int) -> None
    >>> img = cv2.imread('example0/ring.jpg', cv2.IMREAD_COLOR)
    >>> output_encircled(img, 'example0/ring.jpg', 126, 0, 0, dirname='stage1')
    Drawing to example0-stage1/ring.jpg;  radius: 126; centre: (0,0)
    >>> output_encircled(img, 'example0/ring.jpg', 100, 50, 200, dirname='stage1')
    Drawing to example0-stage1/ring.jpg;  radius: 100; centre: (50,200)
    '''
    
    # STAGE 3: output circles of sun and moon
    # draw sun circle on circled
    circled = img.copy()
    a = cv2.circle(circled, (sun_x, sun_y), sun_radius, RED, 4)
    a = cv2.rectangle(circled, (sun_x - 5, sun_y - 5), (sun_x + 5, sun_y + 5), ORANGE, -1)

    if original_moon_x != -1:
        a = cv2.circle(circled, (original_moon_x, original_moon_y), moon_radius, GREEN, 4)
        a = cv2.rectangle(
            circled, (original_moon_x - 5, original_moon_y - 5), (original_moon_x + 5, original_moon_y + 5),
            (0, 128, 255), -1
        )

    write_to_fname = path.replace('/', '-' + dirname + '/')   #params['path'] + params['input_dir'].replace('/', '-' + dirname + ' /') + fname
    print('Drawing to ', write_to_fname, ';  radius: ', sun_radius, '; centre: (', sun_x, ',', sun_y, ')', sep='')
    #print(write_to_fname)
    cv2.imwrite(write_to_fname, circled)


def make_json_filename(sun_path, stage):
    '''(str) -> str
    Create a json filename for get_sun_central_radius.
    >>> make_json_filename('example2/DSC05686.jpg', 'stage1')
    'example2-stage1/DSC05686.json'
    >>> make_json_filename('example2/DSC05688.JPG', 'stage1')
    'example2-stage1/DSC05688.json'
    '''
    return sun_path.replace('/','-' + stage + '/').replace('.jpg', '.json').replace('.JPG', '.json')


def get_sun_central_radius(sun_path, min_threshold = 5, max_threshold = 100, threshold_increment = 5):
    '''(str) -> int
    Given filename path, try a range of thresholds and see what the median radius is.
    >>> get_sun_central_radius('example0/ring.jpg')
    126
    >>> get_sun_central_radius('example1/ex1.jpg')
    109
    >>> get_sun_central_radius('example1/ex3.jpg')
    110
    >>> get_sun_central_radius('example2/DSC05686.jpg')
    141
    >>> get_sun_central_radius('example2/DSC05702.jpg')
    143
    '''
    # to avoid recomputing unneccissarily
    json_filename = make_json_filename(sun_path, 'stage1') 
    data = {}
    if os.path.isfile(json_filename):
        with open(json_filename, 'r') as f:
            data = json.load(f)    
            
    # did we get saved data?
    if data and (str(min_threshold) in data):
        #print('data found')
        radiuses = list(data.values())
    else:
        #print('data not found, output to', json_filename)
        data = {}
        radiuses = []
        for i in range(min_threshold, max_threshold, threshold_increment):
            tentative_radius = get_sun_radius(sun_path, i)
            data[i] = tentative_radius
            radiuses.append(tentative_radius)
        # save results to avoid recomputing
        with open(json_filename, 'w+') as f:
            json.dump(data, f) # note: this converts the keys to strings
        
    r = np.array(radiuses)
    return round(np.median( r[ r > 2 ]))


def get_thresholds_for_radius(sun_path, target_radius, tolerance, min_threshold = 5, max_threshold = 100, threshold_increment = 5):
    '''(str, int, int) -> list
    Given a sun image, return a list of sun thresholds that are within tolerance of the target radius.
    This is used for figuring out a suitable sun threshold in stage 1.
    A value of 2 for tolerance seems to be a generally good value.
    
    >>> get_thresholds_for_radius('example0/ring.jpg', 126, 1)
    [15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95]
    >>> get_thresholds_for_radius('example0/ring.jpg', 126, 2)
    [15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95]
    >>> get_thresholds_for_radius('example1/ex1.jpg', 109, 1)
    [10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85]
    >>> get_thresholds_for_radius('example1/ex3.jpg', 110, 1)
    [35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95]
    >>> get_thresholds_for_radius('example1/ex3.jpg', 110, 2)
    [35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95]
    >>> get_thresholds_for_radius('example2/DSC05686.jpg', 141, 1)
    [40, 45, 50, 55, 60, 65, 70, 75, 80]
    >>> get_thresholds_for_radius('example2/DSC05702.jpg', 143, 1)
    [65, 70, 75, 80, 85, 90, 95]
    >>> get_thresholds_for_radius('example2/DSC05702.jpg', 143, 5)
    [65, 70, 75, 80, 85, 90, 95]
    '''
    # to avoid recomputing unneccissarily
    json_filename = make_json_filename(sun_path, 'stage1') 
    data = {}
    if os.path.isfile(json_filename):
        with open(json_filename, 'r') as f:
            data = json.load(f)    
   
    thresholds = []         
    if data and (str(min_threshold) in data):
        saved_thresholds = np.array(list(data.keys())).astype(int)
        radiuses = np.array(list(data.values())).astype(int)
        # there's probably a nicer way to do this
        for i, v in enumerate(radiuses):
            if abs(v - target_radius) <= tolerance:
                thresholds.append( saved_thresholds[i] )        
    else:     
        for i in range(min_threshold, max_threshold, threshold_increment):
            tentative_radius = get_sun_radius(sun_path, i)
            if abs(tentative_radius - target_radius) <= tolerance:
                thresholds.append(i)
    return thresholds


def get_common_threshold(sun_list, blended_radius, tolerance = 2):
    '''(list of str, int, int) -> int
    Given the list of images, and their blended radius, pick a threshold that is common to them all.
    >>> get_common_threshold(['example0/ring.jpg'], 126, 2)    
    15
    >>> get_common_threshold(['example0/ring.jpg'], 126, 1)    
    15
    >>> get_common_threshold(['example1/ex1.jpg', 'example1/ex3.jpg'], 110, 2)    
    35
    >>> get_common_threshold(['example2/DSC05686.jpg', 'example2/DSC05702.jpg'], 142, 2)
    65
    >>> get_common_threshold(['example2/DSC05686.jpg', 'example2/DSC05702.jpg'], 142, 1)
    65
    >>> get_common_threshold(['example2/DSC05686.jpg', 'example2/DSC05688.jpg', 'example2/DSC05702.jpg'], 141, 2)
    70
    >>> get_common_threshold(['example2/DSC05686.jpg', 'example2/DSC05688.jpg', 'example2/DSC05702.jpg'], 141, 1)
    Error: no common thresholds among the sun list. Rerun with a higher tolerance.
    '''
    common_thresholds = set([])
    for i, image_path in enumerate(sun_list):
        threshold_list = get_thresholds_for_radius(image_path, blended_radius, tolerance)
        if i == 0:
            common_thresholds = set(threshold_list)
        else:
            common_thresholds = common_thresholds.intersection( set(threshold_list) )

    if len(common_thresholds) == 0:
        print('Error: no common thresholds among the sun list. Rerun with a higher tolerance.')
    else:
        return min(common_thresholds)
        

def get_directory(path):
    '''
    str -> str
    >>> get_directory('path/to/a/file.type')
    'path/to/a/'
    '''
    return '/'.join(path.split('/')[:-1]) + '/'
        

def whether_calculate_stage1(data, sun_list):
    '''
    (dict, list) -> bool
    Return false if every element in sun_list is in the data dict.
    >>> whether_calculate_stage1({}, ['example2/DSC05686.jpg', 'example2/DSC05688.jpg', 'example2/DSC05702.jpg'])
    True
    >>> whether_calculate_stage1({"example2/DSC05686.jpg": {"sun_radius": 141, "sun_x": 713, "sun_y": 1098, "sun_threshold": 70}, "example2/DSC05688.jpg": {"sun_radius": 141, "sun_x": 1382, "sun_y": 1165, "sun_threshold": 70}, "example2/DSC05702.jpg": {"sun_radius": 141, "sun_x": 477, "sun_y": 1538, "sun_threshold": 70}}, ['example2/DSC05686.jpg', 'example2/DSC05688.jpg', 'example2/DSC05702.jpg'])
    False
    '''
    for sun_path in sun_list:
        if sun_path not in data:    
            return True
    return False
        
        
def stage_1(sun_list, tolerance = 2):
    '''(list of str, int)
    Given a list of images for calibration, use these images to determine
    a sun radius and sun threshold that work for all of these images.
    Output pictures to a stage1 directory showing red circles for the sun 
    radius for you to manually confirm these work.
    If you get an error, you probably need to increase the tolerance.
    
    >>> stage_1(['example0/ring.jpg'], 2)    
    JSON file found at example0-stage1/stage1n1t2.json
    Drawing to example0-stage1/ring.jpg;  radius: 126; centre: (239,180)
    >>> stage_1(['example0/ring.jpg'], 1)    
    JSON file found at example0-stage1/stage1n1t1.json
    Drawing to example0-stage1/ring.jpg;  radius: 126; centre: (239,180)
    >>> stage_1(['example1/ex1.jpg', 'example1/ex3.jpg'], 2)    
    JSON file found at example1-stage1/stage1n2t2.json
    Drawing to example1-stage1/ex1.jpg;  radius: 110; centre: (238,153)
    Drawing to example1-stage1/ex3.jpg;  radius: 110; centre: (276,132)
    >>> stage_1(['example2/DSC05686.jpg', 'example2/DSC05702.jpg'], 2)
    JSON file found at example2-stage1/stage1n2t2.json
    Drawing to example2-stage1/DSC05686.jpg;  radius: 142; centre: (713,1099)
    Drawing to example2-stage1/DSC05702.jpg;  radius: 142; centre: (477,1539)
    >>> stage_1(['example2/DSC05686.jpg', 'example2/DSC05702.jpg'], 1)
    JSON file found at example2-stage1/stage1n2t1.json
    Drawing to example2-stage1/DSC05686.jpg;  radius: 142; centre: (713,1099)
    Drawing to example2-stage1/DSC05702.jpg;  radius: 142; centre: (477,1539)
    >>> stage_1(['example2/DSC05686.jpg', 'example2/DSC05688.jpg', 'example2/DSC05702.jpg'], 2)
    JSON file found at example2-stage1/stage1n3t2.json
    Drawing to example2-stage1/DSC05686.jpg;  radius: 141; centre: (713,1098)
    Drawing to example2-stage1/DSC05688.jpg;  radius: 141; centre: (1382,1165)
    Drawing to example2-stage1/DSC05702.jpg;  radius: 141; centre: (477,1538)
    >>> stage_1(['example2/DSC05686.jpg', 'example2/DSC05688.jpg', 'example2/DSC05702.jpg'], 1)
    JSON file found at example2-stage1/stage1n3t1.json
    Drawing to example2-stage1/DSC05686.jpg;  radius: 141; centre: (713,1098)
    Drawing to example2-stage1/DSC05688.jpg;  radius: 141; centre: (1382,1165)
    Drawing to example2-stage1/DSC05702.jpg;  radius: 141; centre: (477,1538)
    '''
    # have we already calculated everything?
    stage = 'stage1'
    calculating = True
    json_filename = get_directory(sun_list[0]).replace('/', '-' + stage + '/') + stage + 'n' + str(len(sun_list)) + 't' + str(tolerance) + '.json'
    
    data = {}
    if os.path.exists(get_directory(json_filename)):        
        if os.path.isfile(json_filename):
            print('JSON file found at', json_filename)
            with open(json_filename, 'r') as f:
                data = json.load(f)
    else:
        os.makedirs(get_directory(json_filename))

    calculating = whether_calculate_stage1(data, sun_list)
    if calculating:
        print('No JSON file for stage 1 found. Computing and saving one.')
        # first calculate a radius for each sun picture, and average them
        radiuses = []
        for image_path in sun_list:
            radius = get_sun_central_radius(image_path)
            radiuses.append(radius)
        sun_radius = int(round(np.average([radiuses])))

        # now pick sun threshold that is common to them all
        sun_threshold = int(get_common_threshold(sun_list, sun_radius, tolerance = 2))

        for sun_path in sun_list:
            sun_x, sun_y = get_centre_of_sun(sun_path, sun_threshold, sun_radius)
            data[sun_path] = {'sun_radius': sun_radius, 'sun_x': sun_x, 'sun_y': sun_y, 'sun_threshold': sun_threshold}

        # save data
        with open(json_filename, 'w') as g:
            json.dump(data, g)

    # visualize the data to make it easy to check
    # we want to repaint these pictures in case you want to manually adjust the json file
    for sun_path in sun_list:
        img = cv2.imread(sun_path, cv2.IMREAD_COLOR)
        output_encircled(img, sun_path, data[sun_path]['sun_radius'], data[sun_path]['sun_x'], data[sun_path]['sun_y'], dirname=stage)



def get_amount_of_sun(fname, params, img):
    '''str, dict, CV2 -> float
    How much of the black and white version of img is white?'''
    
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # otsu
    # _, img_binary = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, img_binary = cv2.threshold(img_gray, params['sun_threshold'], 255, cv2.THRESH_BINARY)
    # save sun_binary
    cv2.imwrite(params['path'] + params['input_dir'].replace('/', '-sun-binary/') + fname, img_binary)
    
    # for figuring out percentage of coverage
    all_ones = np.ones(img.shape)
    
    return np.sum(img_binary) / np.sum(all_ones)






def get_centre_of_moon(fname, params, img):
    '''str, dict, CV2 image -> int, int
    Find the centre of the moon, using img and img_gray (from finding centre of sun) as well as radius.
    This does not appear to work properly around the totality.
    '''
    # STAGE 2: FIND CENTRE OF MOON

    moon_mask_r = params['moon_radius'] # sun_mask_r + params['
    moon_mask = np.zeros((moon_mask_r * 2 + 1, moon_mask_r * 2 + 1), np.float32)
    moon_mask = cv2.circle(moon_mask, (moon_mask_r, moon_mask_r), moon_mask_r, 1.0, -1)

    # STAGE 2a: set up canvas. I have no idea how to simplify it, because I have no idea what is happening here.
    sun_mask_r = params['sun_radius'] # 141 for my photos, 110 for OG's # sun radius
    
    canvas_size_original = (sun_mask_r + moon_mask_r * 2) * 2
    img_h, img_w = img.shape[:2]
    canvas_size_alternate = max( img_h, img_w )
    canvas_size = max(canvas_size_alternate, canvas_size_original)
    
    # pad image to canvas_size
    pad_x, pad_y = (canvas_size - img_w) // 2, (canvas_size - img_h) // 2
    padded = np.pad(img, ((pad_y, pad_y), (pad_x, pad_x), (0, 0)))
    # calculate the distance between the center of the sun and the center of the image
    center_x, center_y = img_w // 2, img_h // 2
    canvas = np.zeros((canvas_size, canvas_size, img.shape[2]), dtype=np.uint8)
    dx, dy = center_x - sun_x, center_y - sun_y
    # draw sun at the center of the canvas
    left, right = (0, dx) if dx > 0 else (-dx, 0)
    up, down = (0, dy) if dy > 0 else (-dy, 0)
    pad_h, pad_w = padded.shape[:2]
    canvas[down:(pad_h - up), right:(pad_w - left), :] = padded[up:(pad_h - down), left:(pad_w - right), :]

    # STAGE 2b: look for moon
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # grayscale
    canvas_gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    # otsu
    th, canvas_gray = cv2.threshold(canvas_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    canvas_gray, canvas_binary = cv2.threshold(canvas_gray, th + params['moon_threshold_mod'], 255, cv2.THRESH_BINARY)
    # save moon_binary
    canvas_gray, moon_binary = cv2.threshold(img_gray, th + params['moon_threshold_mod'], 255, cv2.THRESH_BINARY)
    cv2.imwrite(params['path'] + params['input_dir'].replace('/', '-moon-binary/') + fname, moon_binary)

    # STAGE 2c: ???
    # fft convolve
    moon_signal = signal.fftconvolve(canvas_binary.astype('float') / 255, moon_mask, mode='valid')
    # find coordinates of the smallest values in moon_signal that are also inside the moon_signal_mask
    moon_signal_mask_size = canvas_size - moon_mask_r * 2
    moon_signal_mask_r = moon_signal_mask_size // 2
    moon_signal_mask = np.zeros((moon_signal_mask_size, moon_signal_mask_size), np.float32)
    moon_signal_mask = cv2.circle(moon_signal_mask, (moon_signal_mask_r, moon_signal_mask_r), moon_signal_mask_r, 1.0, -1)
    rows, cols = np.where((moon_signal < 1) * (moon_signal_mask > 0))
    # find the coordinate closest to the center of the sun (closest to the center of the moon_signal image)
    moon_signal_center = moon_signal.shape[0] // 2
    moon_x, moon_y, min_dist = 0, 0, 99999
    for i in range(rows.shape[0]):
        dist = spatial.distance.euclidean([cols[i], rows[i]], [moon_signal_center, moon_signal_center])
        if dist < min_dist:
            moon_x, moon_y, min_dist = int(cols[i]), int(rows[i]), float(dist)
    # calculate moon coordinates in the original image
    # moon_x is relative to moon_signal (4802, 4802)
    original_moon_x = moon_x - (moon_signal_mask_size - img_w) // 2 - dx
    original_moon_y = moon_y - (moon_signal_mask_size - img_h) // 2 - dy
    
    #print('Special moon', moon_x, moon_y)
    
    return original_moon_x, original_moon_y





if __name__ == '__main__':
    sun_path = '/home/patitsas/git/solar-eclipse-timelapse-aligner/example1/ex1.jpg'
    r = get_sun_radius(sun_path, 25)
    print(r)
    doctest.testmod()
    '''
    params = {'sun_threshold':75, 'moon_threshold_mod':5, 'sun_radius':110, 'moon_radius':108, 'path':'/home/patitsas/git/solar-eclipse-timelapse-aligner/', 'input_dir':'example2/'}
    
    if params['input_dir'] in ['example1/']:
        params['sun_radius'] = 110
        params['moon_radius'] = 108
    else:
        params['sun_radius'] = 140
        params['moon_radius'] = 137

    input_path = params['path'] + params['input_dir']

    
    for fname in sorted(os.listdir(input_path)):
        jpg_path = input_path + fname
        
        # STAGE 0: read in the input image
        img = cv2.imread(str(jpg_path), cv2.IMREAD_COLOR)

        # STAGE 1: find centre of sun
        perc_sun = get_amount_of_sun(fname, params, img)
        sun_x, sun_y = get_centre_of_sun(fname, params, img)
        # STAGE 2: find centre of moon
        original_moon_x, original_moon_y = get_centre_of_moon(fname, params, img)
        # STAGE 3: output circles of sun and moon
        output_encircled(fname, params, sun_x, sun_y, original_moon_x, original_moon_y)

        # STAGE 4: what to do about totality???
        
        # https://www.geeksforgeeks.org/circle-detection-using-opencv-python/
        # Convert to grayscale. 
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
          
        # Blur using 3 * 3 kernel. 
        gray_blurred = cv2.blur(gray, (3, 3)) 
          
        # Apply Hough transform on the blurred image. 
        detected_circles = cv2.HoughCircles(gray_blurred,  
                           cv2.HOUGH_GRADIENT, 1, 20, param1 = 50, 
                       param2 = 30, minRadius = 1, maxRadius = 1000) 
        # Draw circles that are detected. 
        if detected_circles is not None: 
            # Convert the circle parameters a, b and r to integers. 
            detected_circles = np.uint16(np.around(detected_circles)) 
          
            for pt in detected_circles[0, :]: 
                #print('DC', fname, pt)
                a, b, r = pt[0], pt[1], pt[2] 
          
                # Draw the circumference of the circle. 
                cv2.circle(img, (a, b), r, (0, 255, 0), 2) 
          
                # Draw a small circle (of radius 1) to show the center. 
                cv2.circle(img, (a, b), 1, (0, 0, 255), 3) 
                #cv2.imwrite(params['path'] + params['input_dir'].replace('/', '-test/'), img) 
                #cv2.waitKey(0) 

        print(fname,  perc_sun, sun_x, sun_y, original_moon_x, original_moon_y, math.dist([sun_x, sun_y], [original_moon_x, original_moon_y]))
    '''
    
    '''
    TODO: set this up to work with multiple files
        then set it up so you don't have to re-circle unless specified
        get timestamps from files for making videos
        

    '''

