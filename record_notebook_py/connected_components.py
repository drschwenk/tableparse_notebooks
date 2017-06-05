# -*- coding: utf-8 -*-
# <nbformat>4</nbformat>

# <markdowncell>

# # Table of Contents
# * [Introduction](#Introduction)
# 	* [Guiding questions](#Guiding-questions)
# 	* [Related notebooks](#Related-notebooks)
# * [Setup](#Setup)
# 	* [Imports](#Imports)
# 	* [Load data](#Load-data)
# 	* [code](#code)
# 	* [load](#load)
# * [Analysis](#Analysis)
# 	* [code](#code)
# 	* [run](#run)
# * [Conclusions](#Conclusions)
# 	* [Key findings](#Key-findings)
# 	* [Next steps](#Next-steps)


# <markdowncell>

# # Introduction

# <markdowncell>

# ## Guiding questions

# <markdowncell>

# * Guiding question:
# Can I improve grid extraction by finding largest component first, then extracting contours

# <markdowncell>

# ## Related notebooks

# <markdowncell>

# * **Related notebooks:**  early_table_parse_experiment.ipynb

# <markdowncell>

# # Setup

# <markdowncell>

# ## Imports

# <codecell>

%%capture
from __future__ import division
import numpy as np
import pandas as pd
import scipy.stats as st
import itertools
import math
import hashlib
from collections import Counter, defaultdict
%load_ext autoreload
%autoreload 2

# <codecell>

%%capture
import matplotlib as mpl
import matplotlib.pylab as plt
%matplotlib inline
%load_ext base16_mplrc
%base16_mplrc light solarized
plt.rcParams['grid.linewidth'] = 0
plt.rcParams['figure.figsize'] = (16.0, 10.0)

# <codecell>

import os
import cv2

import PIL.Image as Image
import skimage.filters

from urllib.request import url2pathname

# <codecell>

%load_ext version_information
%reload_ext version_information
%version_information numpy, matplotlib, pandas, scipy, cv2, skimage, PIL

# <markdowncell>

# # Analysis

# <markdowncell>

# ## code

# <codecell>

def ifa(img_arr):
    return Image.fromarray(img_arr)


def ifb(blobs):
    plt.imshow(blobs, cmap='nipy_spectral')
    _ = plt.axis('off')

    
def random_color():
    import random
    return random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)


def draw_detections(img_path, found_cells):    
    image = cv2.imread(img_path)
    color_counter = 0
    for cell in found_cells:
        start_x = cell[0][0]
        start_y = cell[0][1]
        end_x = cell[1][0]
        end_y = cell[1][1]
        cv2.rectangle(image, (start_x, start_y), (end_x, end_y), color=random_color(), thickness=2)
        color_counter += 1
    return Image.fromarray(image)


# <codecell>

def convert_binary_image(binary_image):
    image = binary_image.astype('uint8') * np.ones_like(binary_image) * 255
    return image


def foreground_image(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh_val = skimage.filters.threshold_li(gray)
    foreground = gray < thresh_val
    return convert_binary_image(foreground)


def connect_and_label_components(foreground_img):
    img = foreground_img
    blobs = img > img.mean()
    labeled_image_components = skimage.measure.label(blobs)
    return labeled_image_components


def find_largest_component(labeled_components):
    regions = skimage.measure.regionprops(labeled_components)
    regions_by_area = sorted(regions, key=lambda x: x.convex_area, reverse=True)
    largest_region = regions_by_area[0]
    return convert_binary_image(labeled_components == largest_region.label)


def cell_from_contour(contour):
    cont = contour.reshape((contour.shape[0], 2))
    bounding_box = (cont[:,:1].min(), cont[:,1:].min()) , (cont[:,:1].max(), cont[:,1:].max())
    return bounding_box

def compute_grid_contours(assumed_grid_mask):
    converted_mask =  assumed_grid_mask * np.ones_like(assumed_grid_mask) * 255
    _, contours, hierarchy = cv2.findContours(converted_mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    hierarchy = hierarchy.reshape(hierarchy.shape[1:])
    child_idxs = hierarchy[:,-1] == 0
    return [cont for idx, cont in enumerate(contours) if child_idxs[idx]]

# <markdowncell>

# ## run

# <codecell>

hard_image = '/Users/schwenk/wrk/tableparse/data/tricky_tables/unnamed-3.png'
another_image = '/Users/schwenk/wrk/tableparse/data/tricky_tables/unnamed-2.png'
easy_image = '/Users/schwenk/wrk/tableparse/vision-tableparse/examples/sight-word-bingo.png'

output_dir = 'test_cc_output'
test_images = glob.glob('/Users/schwenk/wrk/tableparse/data/test_data/images/*')

# <codecell>

image_to_do = test_images[100]

# <codecell>

# %%time
# for image_to_do in test_images[97:]:
#     test_img = cv2.imread(image_to_do)
#     image_name = os.path.split(image_to_do)[-1]

#     foreground_img = foreground_image(test_img)
#     connected_components = connect_and_label_components(foreground_img)
#     candidate_grid = find_largest_component(connected_components)
#     grid_contours = compute_grid_contours(candidate_grid)

#     bounding_boxes = [cell_from_contour(cont) for cont in grid_contours]

#     cv2.imwrite(os.path.join(output_dir, image_name), test_img)
#     cv2.imwrite(os.path.join(output_dir, image_name.replace('.png', '_grid.png')), candidate_grid)
#     draw_detections(image_to_do, bounding_boxes).save(os.path.join(output_dir,image_name.replace('.png', '_cells.png')))

# <codecell>

test_img = cv2.imread(image_to_do)
image_name = os.path.split(image_to_do)[-1]

foreground_img = foreground_image(test_img)
connected_components = connect_and_label_components(foreground_img)
candidate_grid = find_largest_component(connected_components)
grid_contours = compute_grid_contours(candidate_grid)

bounding_boxes = [cell_from_contour(cont) for cont in grid_contours]
grid_cells = [GridCell(cell_from_contour(cont)) for cont in grid_contours]

# <codecell>

grid_cells

# <markdowncell>

# next and previous contours at the same hierarchical level, the first child contour and the parent contour

# <codecell>

draw_detections(image_to_do, bounding_boxes)

# <codecell>



# <codecell>

ifa(test_img)

# <codecell>

ifa(foreground_img)

# <codecell>

ifa(candidate_grid)

# <codecell>

ifb(connected_components)

# <markdowncell>

# # match to OCR

# <codecell>

image_to_do = test_images[97]

test_img = cv2.imread(image_to_do)
image_name = os.path.split(image_to_do)[-1]

foreground_img = foreground_image(test_img)
connected_components = connect_and_label_components(foreground_img)
candidate_grid = find_largest_component(connected_components)
grid_contours = compute_grid_contours(candidate_grid)

bounding_boxes = [cell_from_contour(cont) for cont in grid_contours]
draw_detections(image_to_do, bounding_boxes)

# <codecell>

ifa(candidate_grid)

# <codecell>

test_ocr_img =cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY) + candidate_grid

# <codecell>

ifa(test_ocr_img)

# <codecell>

img_buffer = BytesIO()

# <codecell>

ifa(test_ocr_img).save(img_buffer, format="PNG")
img_str = base64.b64encode(img_buffer.getvalue())

# <codecell>

img_str = base64.b64encode(cv2.imencode('.png', test_ocr_img)[1].tostring())

# <codecell>

import base64
from io import BytesIO
import requests

# <codecell>

api_entry_point = 'http://vision-ocr.dev.allenai.org/v1/ocr'
b64_encoded_image = base64.b64encode(test_ocr_img.tostring())
header = {'Content-Type': 'application/json'}
request_data = {
    'image': img_str.decode('utf-8'),
    'mergeBoxes': False,
    'includeMergedComponents': True
}

json_data = json.dumps(request_data)
response = requests.post(api_entry_point, data=json_data, headers=header)
print(response.reason)
json_response = json.loads(response.content.decode())

# <codecell>

def query_vision_ocr(image_url, merge_boxes=False, include_merged_components=False, as_json=True):
    b64_encoded_image = base64.b64encode(test_ocr_img)
    api_entry_point = 'http://vision-ocr.dev.allenai.org/v1/ocr'
    b64_encoded_image = base64.b64encode(test_ocr_img)

    header = {'Content-Type': 'application/json'}
    request_data = {
        'image': base_64_img,
        'mergeBoxes': merge_boxes,
        'includeMergedComponents': include_merged_components
    }

    json_data = json.dumps(request_data)
    response = requests.post(api_entry_point, data=json_data, headers=header)
    print(response.reason)
    json_response = json.loads(response.content.decode())
    if as_json:
        response = json_response
    return response

# <codecell>

class Box(object):
    
    def __init__(self):
        self.upper_x = 0
        self.lower_x = 0
        self.upper_y = 0
        self.lower_y = 0
        self.ocr = None
            
    def __repr__(self):
        return ', '.join(map(str, [self.ocr, self.u_x(), self.l_x(), self.u_y(), self.l_y()]))
        
    def u_x(self):
        return self.upper_x
    
    def l_x(self):
        return self.lower_x
    
    def u_y(self):
        return self.upper_y
    
    def l_y(self):
        return self.lower_y
    
    def area(self):
        return (self.u_x() - self.l_x()) * (self.u_y() - self.l_y())
    
    def text(self):
        return self.ocr
    
    
class OcrBox(Box):
    
    def __init__(self, detection):
        rect = detection['rectangle']
        self.upper_x = rect[1]['x']
        self.lower_x = rect[0]['x']
        self.upper_y = rect[1]['y']
        self.lower_y = rect[0]['y']
        self.ocr = detection['value']
        

        
class GridCell(Box):
    
    def __init__(self, coords):
        self.upper_x = coords[1][0]
        self.lower_x = coords[0][0]
        self.upper_y = coords[1][1]
        self.lower_y = coords[0][1]
        self.ocr = ""
        
    def assign_detection(self, ocr):
        self.ocr = ocr

# <codecell>

bounding_boxes

# <codecell>

def boxes_overlap(detected_box, grid_box, thresh=0.9):
    dx = min(detected_box.u_x(), grid_box.u_x()) - max(detected_box.l_x(), grid_box.l_x())
    dy = min(detected_box.u_y(), grid_box.u_y()) - max(detected_box.l_y(), grid_box.l_y())
    if (dx > 0) and (dy > 0):
        intersection_area = dx * dy
        return float(intersection_area) / min(detected_box.area(), grid_box.area()) > thresh
    else:
        return False
    
    
def assign_ocr_to_cells(detected_boxes, grid_boxes, thresh=0.5):
    for db in detected_boxes:
        for gb in grid_boxes:
            if boxes_overlap(db, gb, thresh):
                gb.assign_detection(db.text())
    return 

# <codecell>

def draw_detections_w_text(img_path, grid_cells):    
    image = cv2.imread(img_path)
    color_counter = 0
    for cell in grid_cells:
        cv2.rectangle(image, (cell.l_x(), cell.l_y()), (cell.u_x(), cell.u_y()), color=random_color(), thickness=2)
        cv2.putText(image, cell.text(), (cell.l_x(), cell.u_y()), cv2.FONT_HERSHEY_PLAIN, 0.8, (0,0,255), 2)
        color_counter += 1
    return Image.fromarray(image)

# <codecell>

ocr_detections = [OcrBox(det) for det in json_response['detections']]

# <codecell>

grid_cells = [GridCell(cell) for cell in bounding_boxes]

# <codecell>

matches = assign_ocr_to_cells(ocr_detections, grid_cells)

# <codecell>

matches

# <codecell>

ocr_detections

# <codecell>

draw_detections_w_text(image_to_do, grid_cells)

# <codecell>



# <markdowncell>

# # Conclusions

# <markdowncell>



# <markdowncell>

# ## Key findings

# <markdowncell>

# * Key finding 1

# <markdowncell>

# ## Next steps

# <markdowncell>

# * Next steps 1
