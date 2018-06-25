import numpy as np
import  tensorflow as tf


class BM:
    def __init__(self,  image_height, image_width, SADWindowSize=15, minDisparity=0, numberOfDisparities=32,
                 textureThreshold=10, uniquenessRatio=15, speckleWindowSize=100,
                 speckleRange=32, disp12MaxDiff=1):
        self.image_height = image_height
        self.image_width = image_width
        self.SADWindowSize = SADWindowSize
        self.minDisparity = minDisparity
        self.numberOfDisparities = numberOfDisparities
        self.textureThreshold = textureThreshold
        self.uniquenessRatio = uniquenessRatio
        self.speckleWindowsSize = speckleWindowSize
        self.speckleRange = speckleRange
        self.disp12MaxDiff = disp12MaxDiff
        self.left_image = tf.placeholder(tf.int32, [self.image_height, self.image_width, 1])
        self.right_image = tf.placeholder(tf.int32, [self.image_height, self.image_width, 1])

    def  cvFindCorrespondenceBM(self):
        win = self.SADWindowSize // 2
        shape = self.left_image.get_shape()
        disp = tf.get_variable('disp', [shape[0] - 1, shape[1] - 1, 1], tf.int32, tf.zeros_initializer)
        for i in range(0, shape[0] - 1 - self.SADWindowSize):
            for j in range(0, shape[1] - 1 - self.SADWindowSize - self.numberOfDisparities):
                bestMatchSoFar = self.coMatch(i, j)
                indices = tf.constant([[(i + win) * self.image_height + (j + win)]])
                updates = tf.constant([bestMatchSoFar])
                disp_shape = tf.constant([self.image_height * self.image_width])
                scatter = tf.reshape(tf.scatter_nd(indices, updates, disp_shape), [shape[0], shape[1], shape[2]])
                disp = tf.add(disp, scatter)
        return disp

    def coMatch(self, i, j):
        prevSad_1 = 4294967295
        prevSad_2 = 4294967295
        bestMatchSoFar = self.minDisparity
        bestMatchSoFar_1 = self.minDisparity
        bestMatchSoFar_2 = self.minDisparity
        for dispRange in range(self.minDisparity, self.numberOfDisparities):
            block_left = tf.image.crop_to_bounding_box(self.left_image, i, j,
                                                       self.SADWindowSize, self.SADWindowSize)
            block_right = tf.image.crop_to_bounding_box(self.right_image, i, j + dispRange,
                                                        self.SADWindowSize, self.SADWindowSize)
            sad = tf.reduce_sum(tf.abs(tf.subtract(block_left, block_right)))
            if prevSad_1 > sad:
                prevSad_1 = sad
                bestMatchSoFar_1 = dispRange
        for dispRange in range(self.minDisparity, self.numberOfDisparities):
            block_left = tf.image.crop_to_bounding_box(self.right_image, i, j + bestMatchSoFar_1,
                                                       self.SADWindowSize, self.SADWindowSize)
            block_right = tf.image.crop_to_bounding_box(self.left_image, i, j + dispRange,
                                                        self.SADWindowSize, self.SADWindowSize)
            sad = tf.reduce_sum(tf.abs(tf.subtract(block_left, block_right)))
            if prevSad_2 > sad:
                prevSad_2 = sad
                bestMatchSoFar_2 = dispRange
        if np.abs(bestMatchSoFar_1, bestMatchSoFar_2) < self.disp12MaxDiff:
            bestMatchSoFar = bestMatchSoFar_1
        return bestMatchSoFar

