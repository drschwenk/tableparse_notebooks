# -*- coding: utf-8 -*-
# <nbformat>4</nbformat>

# <markdowncell>

# # Table of Contents
# * [Alternate tables](#Alternate-tables)
# 	* [pairing images and annotations](#pairing-images-and-annotations)
# 	* [sampling images](#sampling-images)
# * [Regents tables](#Regents-tables)
# * [Hough lines experiment](#Hough-lines-experiment)
# * [End](#End)


# <codecell>

%%capture
from __future__ import division
import numpy as np
import pandas as pd
import scipy.stats as st
import itertools
import math
from collections import Counter, defaultdict
%load_ext autoreload
%autoreload 2

import os
import cv2
import PIL.Image as Image

# <markdowncell>

# # Alternate tables

# <codecell>

image_path_prefix = '../data/small_table_training/'
anno_path_prefix = '../data/exp_nicks_data/table-research/ground_truth/alternate/'

image_files = os.listdir(image_path_prefix)
anno_files = os.listdir(anno_path_prefix)

# <markdowncell>

# ## pairing images and annotations

# <codecell>

image_bases = [''.join(f.split('.')[:-1]) for f in image_files]
anno_bases = [''.join(f.split('.')[:-2]) for f in anno_files]

# <codecell>

images_with_anno  = [f  for f in image_files if ''.join(f.split('.')[:-1]) in anno_bases]

# <codecell>

bases_intersection = set(image_bases).intersection(set(anno_bases))

# <codecell>

len(bases_intersection)

# <codecell>

len(anno_bases)

# <codecell>

len(anno_bases)

# <markdowncell>

# images missing annotations

# <codecell>

set(image_bases[:100]).difference(set(anno_bases))

# <markdowncell>

# ## sampling images

# <codecell>

sample_n = 30

# <codecell>

sample_image = images_with_anno[sample_n]

# <codecell>

Image.open(image_path_prefix + sample_image)

# <codecell>

with open(anno_path_prefix + anno_files[sample_n]) as f:
    sample_anno = f.readlines()

split_lines = [l.split(',', maxsplit=4) for l in sample_anno]

# <codecell>

split_lines[0]

# <markdowncell>

# # Regents tables

# <codecell>

regents_image_path_prefix = '../data/exp_nicks_data/regents_images/'
regents_anno_path_prefix = '../data/exp_nicks_data/regents_anno/'

# <codecell>

regents_anno = os.listdir(regents_anno_path_prefix)

# <codecell>

regents_anno_8th = {an: ".PNG" for an in regents_anno if '_8_' in an}
regents_anno_4th = {an: ".PNG" for an in regents_anno if '_4_' in an}
regents_anno_other = {an: ".PNG" for an in regents_anno if an not in regents_anno_4th and an not in regents_anno_8th}

# <codecell>

# assert(set(regents_anno_other + regents_anno_8th + regents_anno_4th) == set(regents_anno))

# <codecell>

regents_images_4 = os.listdir(regents_image_path_prefix + '/4th')
regents_images_8 = os.listdir(regents_image_path_prefix + '/8th')
regents_images_8 = [ri for ri in regents_images_8 if '2011' in ri]

# <codecell>

regents_images_8

# <codecell>

img_n = 0

# <codecell>

anno_n = 0

# <codecell>

anno_n += 1
with open(regents_anno_path_prefix + list(regents_anno_8th.keys())[anno_n]) as f:
    print(list(regents_anno_8th.keys())[anno_n])
    print()
    print(f.read())

# <codecell>

img_n +=1
print(regents_images_8[img_n])
Image.open(regents_image_path_prefix + '/8th/' + regents_images_8[img_n])

# <codecell>

name_mapping = {
    '2007_4_15.jpg.txt': '2007_4th_Grade_09.PNG',
    '2009_4_31b.jpg.txt': '2009_4th_Grade_11.PNG',
    '2009_4_40.jpg.txt': '2009_4th_Grade_18.PNG',
    '2011_4_32.jpg.txt': '2011_4th_Grade_16.PNG',
    
    
    '2004_8_55_2.jpg.txt': '2004_8th_Grade_53.PNG',
    '2004_8_64-65.jpg.txt': '2004_8th_Grade_55.PNG',
    '2005_8_38.jpg.txt': '2005_8th_Grade_26.PNG',
    '2005_8_46-48.jpg.txt': '2005_8th_Grade_29.PNG',
    '2005_8_79.jpg.txt': '2005_8th_Grade_44.PNG',
    '2007_8_49-50.jpg.txt': '2007_8th_Grade_20.PNG',
    '2007_8_60.jpg.txt': '2007_8th_Grade_27 .PNG',
    '2009_8_33.jpg.txt': '2009_8th_Grade_16.PNG',
    '2009_8_79-81.jpg.txt': '2009_8th_Grade_41.PNG',
    '2009_8_82-83b.jpg.txt': '2009_8th_Grade_43.PNG',
    '2011_8_56.jpg.txt': '2011_8th_Grade_33.PNG',
    '2011_8_79-80.jpg.txt': '.PNG'
}

# <codecell>



# <codecell>



# <codecell>



# <codecell>



# <codecell>



# <markdowncell>

# # Hough lines experiment

# <codecell>

easy_image = '/Users/schwenk/wrk/tableparse/vision-tableparse/examples/example_1.png'

# <codecell>

# img = cv2.imread(image_path_prefix + sample_image)
img = cv2.imread(easy_image)

gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)


edges = cv2.Canny(gray, 100, 200, apertureSize=3, L2gradient=1)

minLineLength = 30
maxLineGap = 10
lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 120, minLineLength=10, maxLineGap=2)
for x in range(0, len(lines)):
    for x1,y1,x2,y2 in lines[x]:
        cv2.line(img,(x1,y1),(x2,y2),(0,255,0),2)

# <codecell>

Image.fromarray(edges)

# <codecell>

lines.shape

# <codecell>

Image.fromarray(img)

# <codecell>

import cv2
import numpy as np
import os.path
from collections import defaultdict


def ik(x, y):
    return '.'.join([str(x), str(y)])


def boxes_from_intersections(image_bw, h_intersections, v_intersections, all_intersections):
    boxes = []
    for x_i, y_i in all_intersections:
        i_key = ik(x_i, y_i)
        nearest_y = 99999999
        nearest_x = 99999999
        found_point = False
        for x_j, y_j in all_intersections:
            j_key = ik(x_j, y_j)
            if x_j > x_i and y_j > y_i and (h_intersections[i_key] & v_intersections[j_key]) and \
               (v_intersections[i_key] & h_intersections[j_key]) and x_j <= nearest_x and y_j <= nearest_y:
                nearest_x = x_j
                nearest_y = y_j
                found_point = True

        if found_point:
            # x, y, width, height, text
            height = nearest_y - y_i
            width = nearest_x - x_i
            avg_color = (np.average(image_bw[y_i:nearest_y, x_i:nearest_x]))
            if (width <= 15 or height <= 15) and avg_color == 0.0:
                continue
            boxes.append((x_i, y_i, width, height, []))

    return boxes


def get_intersections(img, horiz_lines, vert_lines):
    h_intersections = defaultdict(set)
    v_intersections = defaultdict(set)
    all_intersections = set()

    for h_x1, h_y1, h_x2, h_y2 in horiz_lines:
        intersect_set = set()
        for v_x1, v_y1, v_x2, v_y2 in vert_lines:
            if v_x1 >= h_x1 and v_x1 <= h_x2 and v_y1 <= h_y1 and v_y2 >= h_y1:
                i_key = ik(v_x1, h_y1)
                intersect_set.add(i_key)

        if len(intersect_set) > 2:
            for s in intersect_set:
                all_intersections.add(tuple(map(int, s.split('.'))))
                h_intersections[s] = intersect_set

    for v_x1, v_y1, v_x2, v_y2 in vert_lines:
        intersect_set = set()
        for h_x1, h_y1, h_x2, h_y2 in horiz_lines:
            if v_x1 >= h_x1 and v_x1 <= h_x2 and v_y1 <= h_y1 and v_y2 >= h_y1:
                i_key = ik(v_x1, h_y1)
                intersect_set.add(i_key)

        if len(intersect_set) > 2:
            for s in intersect_set:
                all_intersections.add(tuple(map(int, s.split('.'))))
                v_intersections[s] = intersect_set

    return h_intersections, v_intersections, list(all_intersections)

def supress_lines(lines):
    new_lines = []
    for i, line_a in enumerate(lines):
        suppressed = False
        for j, line_b in enumerate(lines):
            if i >= j:
                continue

            if line_a[0] == line_a[2]:
                min_x = min([line_a[1], line_b[1]])
                max_x = max([line_a[3], line_b[3]])
                intersection = min([line_a[3], line_b[3]]) - max([line_a[1], line_b[1]])
                delta = abs(line_a[0] - line_b[0])

            else:
                min_x = min([line_a[0], line_b[0]])
                max_x = max([line_a[2], line_b[2]])
                intersection = min([line_a[2], line_b[2]]) - max([line_a[0], line_b[0]])
                delta = abs(line_a[1] - line_b[1])

            if intersection > 0 and (intersection/float(max_x - min_x)) > 0.5 and delta < 8:
                suppressed = True
                break

        if not suppressed:
            new_lines.append(line_a)

    return new_lines

# <codecell>

def get_boxes(image_name, base_path):
    horiz_lines = []
    vert_lines = []
    img = cv2.imread(os.path.join(base_path, image_name))
    #img =  cv2.resize(img,(2*img.shape[1], 2*img.shape[0]), interpolation = cv2.INTER_CUBIC)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    (thresh, im_bw) = cv2.threshold(gray, 128, 255, cv2.THRESH_OTSU)

    edges = cv2.Canny(gray, 50, 250, apertureSize=3)
#     edges = cv2.Canny(gray, 100, 200, apertureSize=3, L2gradient=1)
#     return Image.fromarray(edges)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 200, minLineLength=20, maxLineGap=3)
#     lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 120, minLineLength=100, maxLineGap=2)
    if lines is None:
        lines = []

    for info in lines:
        x1, y1, x2, y2 = info[0]
        if y2 < y1:
            y1 = info[0][3]
            y2 = info[0][1]

        # horizontal line
        offsets = [-1, 0, 1]
        if y1 - y2 == 0:
            avg_above = avg_below = 256
            avg_center = np.average(gray[y1:y2 + 1, x1:x2 + 1])

            if y1 > 0:
                avg_above = np.average(gray[y1 - 1:y2, x1:x2 + 1])

            if y2 + 1 < gray.shape[0]:
                avg_below = np.average(gray[y1 + 1:y2 + 2, x1:x2 + 1])

            # assuming black lines, could do something to check for background color

            # this occurs from edges detected in gray areas that aren't cell boundaries
            if np.min([avg_above, avg_center, avg_below]) > 192:
                continue

            y1 += offsets[np.argmin([avg_above, avg_center, avg_below])]

            y2 = y1

            while x2 + 1 < im_bw.shape[1] and abs(im_bw[y1:y2 + 1, x2 + 1:x2 + 2][0,0] - np.average(im_bw[y1:y2 + 1, x1:x2 + 1])) < 16:
                x2 += 1

            while x1 > 0 and abs(im_bw[y1:y2 + 1, x1 - 1:x1][0,0] - np.average(im_bw[y1:y2 + 1, x1:x2 + 1])) < 16:
                x1 -= 1

            horiz_lines.append((x1, y1, x2, y2))
        elif x1 - x2 == 0:
            avg_right = avg_left = 256
            avg_center = np.average(gray[y1:y2 + 1, x1:x2 + 1])

            if x1 > 0:
                avg_left = np.average(gray[y1:y2 + 1, x1 - 1:x2])

            if x2 + 1 < gray.shape[1]:
                avg_right = np.average(gray[y1:y2 + 1, x1 + 1: x2 + 2])

            x1 += offsets[np.argmin([avg_left, avg_center, avg_right])]

            x2 = x1

            while y2 + 1 < im_bw.shape[0] and abs(im_bw[y2 + 1:y2 + 2, x1:x2 + 1][0,0] - np.average(im_bw[y1:y2 + 1, x1:x2 + 1])) < 16:
                y2 += 1

            while y1 > 0 and abs(im_bw[y1 - 1:y1, x1:x2 + 1][0,0] - np.average(im_bw[y1:y2 + 1, x1:x2 + 1])) < 16:
                y1 -= 1

            vert_lines.append((x1, y1, x2, y2))
    horiz_lines = supress_lines(horiz_lines)
    vert_lines = supress_lines(vert_lines)

    sorted_h_lines = sorted(horiz_lines, key=lambda l: l[1])
    sorted_v_lines = sorted(vert_lines, key=lambda l: l[0])
    h_intersections, v_intersections, all_intersections = get_intersections(img, sorted_h_lines, sorted_v_lines)
    return boxes_from_intersections(im_bw, h_intersections, v_intersections, all_intersections)

# <codecell>

def random_color():
    import random
    return random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)

def draw_detections(img_path, found_cells):
    colors = [
        (255, 0, 0),
        (0, 255, 0),
        (0, 0, 255),
        (255, 255, 0),
        (0, 255, 255),
        (255, 0, 255),

        (128, 0, 0),
        (0, 128, 0),
        (0, 0, 128),
        (128, 128, 0),
        (0, 128, 128),
        (128, 0, 128),

        (255, 128, 0),
        (0, 128, 255),
        (128, 255, 0),
        (0, 255, 128),
        (255, 0, 128),
        (128, 0, 255)]
    
    image = cv2.imread(img_path)
    color_counter = 0
    for cell in found_cells:
        start_x = cell[0] 
        start_y = cell[1]
        end_x = cell[0] + cell[2]
        end_y = cell[1] + cell[3]
        cv2.rectangle(image, (start_x, start_y), (end_x, end_y), color=random_color(), thickness=2)
        color_counter += 1
    return Image.fromarray(image)

# <codecell>

old_boxes = get_boxes(sample_image, image_path_prefix)

# <codecell>

new_boxes = get_boxes(sample_image, image_path_prefix)

# <codecell>

len(new_boxes)

# <codecell>

import random

# <codecell>

draw_detections(image_path_prefix + sample_image, random.sample(new_boxes, 10))

# <codecell>



# <markdowncell>

# # End

# <codecell>


