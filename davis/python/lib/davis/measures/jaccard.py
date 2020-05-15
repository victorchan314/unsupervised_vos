# ----------------------------------------------------------------------------
# A Benchmark Dataset and Evaluation Methodology for Video Object Segmentation
#-----------------------------------------------------------------------------
# Copyright (c) 2016 Federico Perazzi
# Licensed under the BSD License [see LICENSE for details]
# Written by Federico Perazzi
# ----------------------------------------------------------------------------

""" Compute Jaccard Index. """

import numpy as np

def db_eval_iou(annotation,segmentation):

	""" Compute region similarity as the Jaccard Index.

	Arguments:
		annotation   (ndarray): binary annotation   map.
		segmentation (ndarray): binary segmentation map.

	Return:
		jaccard (float): region similarity

 """

        print("HELO")
        print(annotation.shape)
        print(segmentation.shape)
        print(annotation)
        print(segmentation)
	annotation   = annotation.astype(np.bool)
	segmentation = segmentation.astype(np.bool)
        print("HELO")
        print(1/0)

	if np.isclose(np.sum(annotation),0) and np.isclose(np.sum(segmentation),0):
		return 1
	else:
		return np.sum((annotation & segmentation)) / \
				np.sum((annotation | segmentation),dtype=np.float32)
