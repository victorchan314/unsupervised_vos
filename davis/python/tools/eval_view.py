#!/usr/bin/env python

# ----------------------------------------------------------------------------
# A Benchmark Dataset and Evaluation Methodology for Video Object Segmentation
#-----------------------------------------------------------------------------
# Copyright (c) 2016 Federico Perazzi
# Licensed under the BSD License [see LICENSE for details]
# Written by Federico Perazzi
# ----------------------------------------------------------------------------

""" Read and display evaluation from HDF5.

EXAMPLE:
	python tools/eval_view.py method.h5

"""


import os
import time
import argparse

import numpy   as np
import os.path as osp

from prettytable import PrettyTable as ptable
from davis.dataset import *
from davis import log

def parse_args():
	"""Parse input arguments."""

	parser = argparse.ArgumentParser(
			description='Read and display evaluation from HDF5.')

	parser.add_argument(dest='input',default=None,type=str,
			help='Path to the HDF5 evaluation file to be displayed.')

	parser.add_argument('--eval_set',default='val',type=str,
			choices=['train','val','trainval'],help='Select set of videos to evaluate.')

	parser.add_argument('--summary',action='store_true',
			help='Print dataset average instead of per-sequence results.')

	# Parse command-line arguments
	args       = parser.parse_args()
	args.input = osp.abspath(args.input)

	return args

if __name__ == '__main__':

	args = parse_args()

	technique = osp.splitext(osp.basename(args.input))[0]

        sequence = None
        if args.eval_set == "val":
            sequence = ['blackswan', 'bmx-trees', 'breakdance', 'camel', 'car-roundabout', 'car-shadow', 'cows', 'dance-twirl', 'dog', 'drift-chicane', 'drift-straight', 'goat', 'horsejump-high', 'kite-surf', 'libby', 'motocross-jump', 'paragliding-launch', 'parkour', 'scooter-black', 'soapbox']
        else:
            raise ValueError("Eval set must be val, not {}".format(args.eval_set))

	db_eval_dict = db_read_eval(technique,raw_eval=False,
			inputdir=osp.dirname(args.input),
                        sequence=sequence)

	log.info("Displaying evaluation of: %s"%osp.basename(args.input))

	db_eval_view(db_eval_dict,
			technique,args.summary,args.eval_set,
                        sequence=sequence)
