from dataclasses import dataclass

import cv2
import numpy as np

from opsi.manager.manager_schema import Function
from opsi.manager.types import RangeType, Slide
from opsi.util.cv.mat import Mat, MatBW

__package__ = "opsi.blobs"
__version__ = "0.123"


class Parameters(Function):
    @dataclass
    class Settings:
        filterByArea: bool
        filterByCircularity: bool
        filterByConvexity: bool
        filterByInertia: bool
        maxArea: int
        minArea: int
        maxCircularity: int
        minCircularity: int
        maxConvexity: int
        minConvexity: int
        maxInertiaRatio: int
        minInertiaRatio: int
        maxThreshold: int
        minThreshold: int
        minDistBetweenBlobs: int

    @dataclass
    class Outputs:
        params:

    def run(self):
        params = Mat.create_blob_params(self.settings.filterByArea, self.settings.filterByCircularity, self.settings.filterByConvexity,
                               self.settings.filterByInertia, self.settings.maxArea, self.settings.minArea, self.settings.maxCircularity,
                               self.settings.minCircularity, self.settings.maxConvexity, self.settings.minConvexity,
                               self.settings.maxInertiaRatio, self.settings.minInertiaRatio, self.settings.maxThreshold,
                               self.settings.minThreshold, self.settings.minDistBetweenBlobs)
        return self.Outputs(params=params)


class BlobDetector(Function):
    @dataclass
    class Inputs:
        img: Mat
        params: cv2.SimpleBlobDetector_Params

    @dataclass
    class Outputs:
        keyPoints: cv2.KeyPoint

    def run(self, inputs):
        keypoints = inputs.img.simple_blob_detection(params=inputs.params)
        return self.Outputs(keyPoints=keypoints)
