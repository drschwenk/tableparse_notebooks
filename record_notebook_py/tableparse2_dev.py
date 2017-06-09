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

# <codecell>

#interactive tableparser dev

# <markdowncell>

# # Setup

# <markdowncell>

# ## Imports

# <codecell>

%%capture
# from __future__ import division
import numpy as np
import pandas as pd
# import scipy.stats as st
# import itertools
# import math
# import hashlib
# from collections import Counter, defaultdict
# %autoreload

import matplotlib as mpl
import matplotlib.pylab as plt
%matplotlib inline
%load_ext base16_mplrc
%base16_mplrc light solarized
plt.rcParams['grid.linewidth'] = 0
plt.rcParams['figure.figsize'] = (16.0, 10.0)

import os
import cv2
import glob
import PIL.Image as Image
# import skimage.filters

# from urllib.request import url2pathname

# <codecell>

%load_ext version_information
%reload_ext version_information
%version_information numpy, matplotlib, pandas, scipy, cv2, skimage, PIL

# <codecell>

hard_image = '/Users/schwenk/wrk/tableparse/data/tricky_tables/unnamed-3.png'
another_image = '/Users/schwenk/wrk/tableparse/data/tricky_tables/unnamed-2.png'
easy_image = '/Users/schwenk/wrk/tableparse/vision-tableparse/examples/sight-word-bingo.png'

output_dir = 'test_cc_output'
test_images = glob.glob('/Users/schwenk/wrk/tableparse/data/test_data/images/*')

# <codecell>

image_to_do = test_images[100]

# <markdowncell>

# # Analysis

# <codecell>

%load_ext autoreload
%autoreload 2

# <codecell>

# ai2.vision.tableparse.img_utils.ifp(image_to_do)

# <codecell>

import ai2.vision.tableparse
# import ai2.vision.tableparse.img_utils

# <codecell>

# image_to_do = '/Users/schwenk/wrk/tableparse/data/test_data/images/table_103.png'
# image_to_do = '/Users/schwenk/wrk/tableparse/data/test_data/images/table_101.png'
image_to_do = '/Users/schwenk/wrk/tableparse/data/test_data/images/table_111.png'

# <codecell>

Image.open(image_to_do)

# <codecell>

import ai2.vision.tableparse.img_utils as imgt

# <codecell>

parsed_table = ai2.vision.tableparse.detect(image_to_do)

# <codecell>

parsed_table.outside_text

# <codecell>

test_ocr_cell = parsed_table.cell_array[0]

# <codecell>

parsed_table.nearest_grid_cell(test_ocr_cell)

# <codecell>



# <codecell>

from ai2.vision.tableparse.cell import GridCell
from ai2.vision.tableparse.table import Table

test_table = Table(parsed_table.cells)

# <codecell>

test_table.cell_array.shape

# <codecell>

test_cell_sort = parsed_table.v_sort()[:15]

# <markdowncell>

# compare gt table shapes

# <codecell>

with open('../data/test_data/table_ground_truth.json', 'r') as f:
    gt_ds = json.load(f)

# <codecell>

dist = np.linalg.norm(np.array([5, 2]) - ([3, 5]))

# <codecell>

dist

# <codecell>

img_path_base = '/Users/schwenk/wrk/tableparse/data/test_data/images/'
parsed_tables = {}
for gtk in sorted(gt_ds.keys())[100:120]:
    gt_anno = gt_ds[gtk]
    print(gt_anno['imageName'])
    image = os.path.join(img_path_base, gt_anno['imageName'])
    parsed_tables[gtk] = ai2.vision.tableparse.detect(image)

# <codecell>

for tk in parsed_tables:
    print(len(gt_ds[tk]['annotations']), np.product(parsed_tables[tk].cell_array.shape), gt_ds[tk]['imageName'])

# <codecell>


