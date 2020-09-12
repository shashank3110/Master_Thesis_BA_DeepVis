#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import nibabel as nib


def read(path='', b_reorient=False, orientation=(('R', 'L'), ('P', 'A'), ('I', 'S'))):
    """
     Read the nifti file and switch to a given orientation

     orientation defaults to std LAS (radiological) - RAS (neurological)
    """

    if not os.path.isfile(path):
        raise ValueError('Provided path is not a valid file')

    image_nii = nib.load(path)
    if b_reorient:
        # switch to given orientation (http://nipy.org/nibabel/image_orientation.html)
        axcodes = nib.aff2axcodes(image_nii.affine)
        orientations = nib.orientations.axcodes2ornt(axcodes, orientation)
        image = image_nii.get_data()
        image = nib.apply_orientation(image, orientations)
        header = image_nii.header
        img_shape = image.shape
        #print(img_shape)
        # image_nii = nib.as_closest_canonical(image_nii)  # quick way to switch to RAS
    else:
        #print('inside orient false')
        image = image_nii.get_data()
        img_shape = image.shape
        #print(img_shape)
    header = image_nii.header
    print(f'preprocessed data shape={img_shape}')
    return image, header, img_shape

