#!/usr/bin/env python
# -------------------------------------------------------------------------------
# A Benchmark Dataset and Evaluation Methodology for Video Object Segmentation
# Copyright (c) 2016 Federico Perazzi
# Licensed under the BSD License [see LICENSE for details]
# Written by Federico Perazzi
# ------------------------------------------------------------------------------

__author__ = 'federico perazzi'
__version__ = '1.0.0'

########################################################################
#
# Interface for accessing the DAVIS dataset.
#
# DAVIS is a video dataset designed for segmentation. The API implemented in
# this file provides functionalities for loading, parsing and visualizing
# images and annotations available in DAVIS. Please visit
# [https://graphics.ethz.ch/~perazzif/davis] for more information on DAVIS,
# including data, paper and supplementary material.
#
# The following API functions are defined:
#	DAVISSegmentationLoader - Class that loads DAVIS data.
#		images		 - return input images.
#		masks			 - return segmentation masks.
#		iternames  - return iterator over tuples of image and mask filenames.
#		iteritems  - return iterator over tuples of images and masks.
#
#	DAVISAnnotationLoader - Class that loads DAVIS annotations and perform evaluation.
#		eval			- perform evaluation of region similarity (J),
#								boundary accuracy (F) and temporal stability (T)
########################################################################

import os
import copy
import skimage.io
import numpy as np
import torch
import torchvision.transforms as transforms
import cv2

from davis import log
from davis.measures import db_eval_boundary,db_eval_iou,db_eval_t_stab

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor()]) 

def resize(img, size=224):
    maximum = np.max(img)
    ndim = img.ndim
    dtype = img.dtype
    if ndim == 2:
        img = img[..., np.newaxis]

    img = img / maximum * 255

    img_tensor = torch.from_numpy(np.transpose(img, axes=(2, 0, 1))).to(torch.uint8)
    img_transformed = transform(img_tensor)
    img = np.transpose(255 * img_transformed.cpu().detach().numpy(), axes=(1, 2, 0)).astype(dtype)
    img = img / 255 * maximum

    if ndim == 2:
        img = img[..., 0]

    return img

def _load_annotation(fname,img_num=0):
	return skimage.io.imread(fname,as_grey=True)

def _load_annotation_resize(fname,img_num=0):
	img = _load_annotation(fname,img_num=img_num)
        print("ENTERED")
        boo = False
        print(img.max())
        if img.max() > .10 and img.max() < .2:
            boo = True
            cv2.imwrite("TEst.png", 255*img)
        img = resize(img)
        print(img.max())
        print("EXIT")
        if boo:
            print(1/0)

        return img

def _load_resize(fname,img_num=0):
	img = skimage.io.imread(fname)
        img = resize(img)

        return img

class DAVISSegmentationLoader(object):
	""" Helper class for accessing the DAVIS dataset.

	Arguments:
		cfg: configuration file provided in davis.config
		sequence (string): sequence name
		masks_dir(string): path to segmentation images.
		ext_im   (string): images file extension
		ext_an   (string): annotations file extension
		load_func        : function to load annotations

	Functions:
		eval       : evaluate sequence
		images		 : return input images
	  masks			 : return segmentation masks
	  iternames  : return iterator over tuples of image and mask filenames
		iteritems  : return iterator over tuples of images and masks

	"""
	def __init__(self,cfg,sequence,masks_dir=None,ext_im="jpg",
			ext_an="png", load_func=_load_annotation):

		super(DAVISSegmentationLoader, self).__init__()

		self._cfg				= cfg
		self.name				= sequence

		self._ext_an		 = ext_an
		self._ext_im		 = ext_im
		self._load_func  = load_func

		self.images_dir = os.path.join(
				self._cfg.PATH.SEQUENCES_DIR,self.name)

		if masks_dir is None:
			self.masks_dir = os.path.join(self._cfg.PATH.ANNOTATION_DIR,self.name)
		else:
			self.masks_dir = os.path.join(masks_dir,self.name)

		assert os.path.isdir(self.images_dir),\
				"Couldn't find folder: %s"%self.images_dir

		#########################################
		# LOAD IMAGES AND MASKS
		#########################################
		self._images = skimage.io.ImageCollection(self.images_dir+"/*%s"%self._ext_im, load_func=_load_resize)

		self._masks = skimage.io.ImageCollection(self.masks_dir+"/*%s"%self._ext_an,
				load_func=_load_annotation_resize)

		#assert len(self._masks) != 0 and len(self._images) != 0
		masks_frames = map(lambda fn:
				int(os.path.splitext(os.path.basename(fn))[0]),self._masks.files)

		# Loading the ground-truth annotations
		if masks_dir is None:
			assert len(self._images) == len(self._masks)
		else:
			assert masks_frames[0] == 0 or masks_frames[0] == 1 and \
					len(masks_frames) == masks_frames[-1] - masks_frames[0] + 1

			self._images = self._images[masks_frames[0]:masks_frames[-1]+1]

		self._frames = list(range(masks_frames[0],
			len(self._images)+masks_frames[0]))

		assert len(self._frames) == len(self._masks) == len(self._images)

		# Compute bounding boxes
		self._bbs = []
		for mask in self._masks:
			coords = np.where(mask!=0)
			if len(coords[0]) <=1:
				self._bbs.append(None)
			else:
				tl = np.min(coords[1]),np.min(coords[0])
				br = np.max(coords[1]),np.max(coords[0])

				self._bbs.append((tl[0],tl[1],br[0],br[1]))

		# FINAL SANITY CHECK
		image_frames = map(lambda fn:
				int(os.path.splitext(os.path.basename(fn))[0]),self._images.files)

		assert np.allclose(image_frames,masks_frames)


	def __getitem__(self, n):
		"""
		Return selected image(s) in the collection.
		Loading is done on demand.
		"""

		if hasattr(n, '__index__'):
			n = n.__index__()

		if type(n) not in [int, slice]:
		 raise TypeError('slicing must be with an int or slice object')

		if type(n) is int:
			n = slice(n,n+1) # Cast to slice

		fidx = range(len(self))[n]

		# A slice object was provided.
		new_ic				 = copy.copy(self)
		new_ic._masks  = self._masks[n]
		new_ic._images = self._images[n]

		new_ic._frames = [self._frames[i] for i in fidx]

		return new_ic

	def __len__(self):
		return len(self._images)

	def __str__(self):
		return self.name

	@property
	def masks(self):
		return self._masks

	@property
	def images(self):
		return self._images

	#######################################
	# ITERATORS
	#######################################
	def iternames(self):
		for im,ma in zip(self.images.files,self._masks.files):
			yield im,ma

	def iteritems(self):
		for im,ma in zip(self._images,self._masks):
			yield im,ma

class DAVISAnnotationLoader(DAVISSegmentationLoader):

	""" Helper class for accessing the DAVIS dataset.

	Arguments:
		cfg: configuration file provided in davis.config
		sequence (string): sequence name
		ext_im   (string): images file extension
		ext_an   (string): annotations file extension
		load_func        : function to load annotations

	Functions: (see DAVISSegmentationLoader documentation)
		eval: evaluate sequence

	"""

	def __init__(self,cfg,sequence,ext_im="jpg",
			ext_an="png",load_func=_load_annotation):

		super(DAVISAnnotationLoader, self).__init__(
				cfg,sequence,None,ext_im,ext_an,load_func)

	def _eval(self,db_segmentation,eval_func,measure,scale=1):
		annotations = self._masks[1:-1]

		# Strip of first and last frame if available
		segmentation = db_segmentation._masks[
				1-db_segmentation._frames[0]:len(annotations)+1-db_segmentation._frames[0]]

		assert len(annotations) == len(segmentation)

		if measure == 'T':
			magic_number = 5.0
			X = np.array([np.nan]+[eval_func(an,sg)*magic_number for an,sg
				in zip(segmentation[:-1],segmentation[1:])] + [np.nan])
		else:
			X = np.array([np.nan]+[eval_func(an,sg) for an,sg
					in zip(annotations,segmentation)] + [np.nan])

		from utils import db_statistics
		M,O,D = db_statistics(X)

		if measure == 'T': O = D = np.nan

		return X,M,O,D

	def eval(self,db_segmentation,measure='J'):

		""" Evaluate sequence.

		Arguments:
			db_segmentation (DAVISSegmentationLoader) : sequence file to be evaluated.
			measure         (string: 'J','F','T')     : measure to be computed

		Returns:
			X: per-frame measure evaluation.
			M: mean   of per-frame measure.
			O: recall of per-frame measure.
			D: decay  of per-frame measure.

		"""

		if measure == 'J':
			return self._eval(db_segmentation,db_eval_iou,measure)
		elif measure=='F':
			return self._eval(db_segmentation,db_eval_boundary,measure)
		elif measure=='T':
			return self._eval(db_segmentation,db_eval_t_stab,measure)
		else:
			raise Exception("Unknown measure=[%s]. Valid options are measure={J,F,T}"%measure)
