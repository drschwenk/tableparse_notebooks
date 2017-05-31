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
# * [Analysis](#Analysis)
# * [Conclusions](#Conclusions)
# 	* [Key findings](#Key-findings)
# 	* [Next steps](#Next-steps)


# <markdowncell>

# # Introduction

# <markdowncell>

# This notebook is meant to reproduce the results reported on the alternate and regents table's in Nick's thesis

# <markdowncell>

# ## Guiding questions

# <markdowncell>

# * Can I reproduce Nick's error distance results on the alternate set
# * Can I get comparable groundtruth comparison for at least some of the Regents set

# <markdowncell>

# ## Related notebooks

# <markdowncell>

# * **nicks_dataset**

# <markdowncell>

# <div class="alert alert-info"> Use this box for important points, reminders</div>

# <markdowncell>

# # Setup

# <markdowncell>

# ## Imports

# <codecell>

%%capture
from __future__ import division
import numpy as np
import pandas as pd
import sklearn as sk
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
import requests
import base64
import PIL.Image as Image
import cv2
from urllib.request import url2pathname
import ai2.vision.utils as ai2vu

# <codecell>

%load_ext version_information
%reload_ext version_information
%version_information numpy, matplotlib, pandas, scipy, sklearn

# <markdowncell>

# ## Load data 

# <markdowncell>

# * **data files:table_ground_truth.json** 
# * **Description of data: Version of Nick's dataset I was able to recover, clean, and consolidate** 

# <codecell>

def compute_data_hash(data_path):
    with open(data_path, 'rb') as f:
        md5_hash = hashlib.md5(f.read()).hexdigest()
        print(data_path, md5_hash)
    return md5_hash

def compute_and_write_hashes(data_files, json_filename):
    data_hashes = {dfp: compute_data_hash(dfp) for dfp in data_files}
    output_dir = 'data_hashes'
    try:
        os.mkdir(output_dir)
    except FileExistsError:
        pass
    
    with open(os.path.join(output_dir, json_filename), 'w') as f:
        json.dump(data_hashes, f)

# <codecell>

data_root_path = '/Users/schwenk/wrk/tableparse/data/test_data/'
data_file_1 = 'table_ground_truth.json'
data_path_1 = os.path.join(data_root_path, data_file_1)
data_file_paths = [data_path_1, ]

# <codecell>

%%javascript
var kernel = IPython.notebook.kernel;
var window_name = window.location.href;
var command = "notebook_url = \"" + escape(window_name) + "\""
kernel.execute(command);

# <codecell>

nb_filename = url2pathname(notebook_url).split('/')[-1]
if nb_filename.endswith('#'):
    nb_filename = nb_filename[:-1]
json_filename = nb_filename.replace('ipynb', 'json')

compute_and_write_hashes(data_file_paths, json_filename)
print('on date:')
!date

# <codecell>

with open(data_path_1) as f:
    table_grountruth = json.load(f)

# <markdowncell>

# # Analysis

# <markdowncell>

# ## Nick's reported results

# <markdowncell>

# ![Screen%20Shot%202017-05-31%20at%2010.16.52%20AM.png](attachment:Screen%20Shot%202017-05-31%20at%2010.16.52%20AM.png)

# <markdowncell>

# ## Running Alternate through current pipeline

# <codecell>

# api_entry_point = 'http://vision-tableparse.dev.allenai.org/v1/tableparse'
api_entry_point_local = 'http://127.0.0.1:5000/v1/tableparse'
header = {'Content-Type': 'application/json'}

def request_table_parse(image, rq_attr):
    request_data = {
        rq_attr: image.decode('utf-8'),
        'border_table': 'True'
    }
    json_data = json.dumps(request_data)
    response = requests.post(api_entry_point_local, data=json_data, headers=header)
    json_response = json.loads(response.content.decode())
    if json_response:
        response = json_response
    return response

def parse_table_image(image_path):
    b64_encoded_image = base64.b64encode(open(image_path, 'rb').read())
    parser_response = request_table_parse(b64_encoded_image, 'image')
    return parser_response

# <codecell>

image_base_path = '/Users/schwenk/wrk/tableparse/data/test_data/images'

# <codecell>

test_table = list(table_grountruth.values())[9]

# <codecell>

img_path = os.path.join(image_base_path, test_table['imageName'])

# <codecell>

Image.open(img_path)

# <codecell>

test_gt_box = test_table['annotations'][0]

# <codecell>

test_gt_box

# <codecell>

test_response = parse_table_image(img_path)

# <codecell>

test_response.keys()

# <codecell>

build_detected_boxes = [item for sublist in list(test_response.values()) for item in sublist]
detected_boxes = [item for sublist in build_detected_boxes for item in sublist]

# <codecell>

detected_boxes[0]

# <codecell>

test_gt_box

# <markdowncell>

# ## cell-level agreement

# <codecell>

def boxes_overlap(detected_box, gt_box, thresh=0.5):
    detected_coords = [[detected_box['rectangle'][n]['x'], detected_box['rectangle'][n]['y']] for n in range(2)]
    detected_coords = [item for sublist in detected_coords for item in sublist]
    gt_coords = gt_box['rectangle']
    gt_coords = gt_coords[:2] + list(map(np.add, gt_coords[:2], gt_coords[2:]))

    def area(box):
        return (box[3] - box[1]) * (box[2] - box[0])

    dx = min(detected_coords[2], gt_coords[2]) - max(detected_coords[0], gt_coords[0])
    dy = min(detected_coords[3], gt_coords[3]) - max(detected_coords[1], gt_coords[1])
    if (dx >= 0) and (dy >= 0):
        intersection_area = dx * dy
        return float(intersection_area) / min(area(detected_coords), area(gt_coords))
    else:
        return False

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
