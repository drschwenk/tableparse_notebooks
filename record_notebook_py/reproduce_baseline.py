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
from urllib.request import url2pathname

# <codecell>

%load_ext version_information
%reload_ext version_information
%version_information numpy, matplotlib, pandas, scipy

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
