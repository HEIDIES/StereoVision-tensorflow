import numpy as np
import tensorflow as tf


class BM:
    def __init__(self,  image_height, image_width, WindowSize=15, minDisparity=0, numberOfDisparities=32,
                 textureThreshold=10, uniquenessRatio=15, speckleWindowSize=100,
                 speckleRange=32, preFilterCap = 31, disp12MaxDiff=1, diffMethod='SAD'):
        self.image_height = image_height
        self.image_width = image_width
        self.WindowSize = WindowSize
        self.minDisparity = minDisparity
        self.numberOfDisparities = numberOfDisparities
        self.textureThreshold = textureThreshold
        self.uniquenessRatio = uniquenessRatio
        self.speckleWindowsSize = speckleWindowSize
        self.speckleRange = speckleRange
        self.preFilterCap = preFilterCap
        self.disp12MaxDiff = disp12MaxDiff
        self.left_image_raw = tf.placeholder(tf.int32, [self.image_height, self.image_width, 1])
        self.right_image_raw = tf.placeholder(tf.int32, [self.image_height, self.image_width, 1])
        self.diffMethod = diffMethod
        self.left_image, self.right_image = self.prefilterXSobel()

    def prefilterXSobel(self):
        weights = tf.Variable(tf.constant([[-1., 0., 1.],
                                           [-2., 0., 2.],
                                           [-1., 0., 1.]]))
        weights = tf.reshape(weights, [3, 3, 1, 1])
        left_image = tf.reshape(tf.cast(self.left_image_raw, tf.float32),
                                     [1, self.image_height, self.image_width, 1])
        right_image = tf.reshape(tf.cast(self.right_image_raw, tf.float32),
                                        [1, self.image_height, self.image_width, 1])
        left_image = tf.nn.conv2d(left_image, weights, [1, 1, 1, 1], padding='SAME')
        right_image = tf.nn.conv2d(right_image, weights, [1, 1, 1, 1], padding='SAME')
        left_image = tf.where(tf.greater(left_image, -self.preFilterCap),
                                   tf.where(tf.greater(left_image, self.preFilterCap),
                                            2. * self.preFilterCap * tf.ones_like(left_image),
                                            left_image + self.preFilterCap), 0. * left_image)
        right_image = tf.where(tf.greater(right_image, -self.preFilterCap),
                                   tf.where(tf.greater(right_image, self.preFilterCap),
                                            2. * self.preFilterCap * tf.ones_like(right_image),
                                            right_image + self.preFilterCap), 0. * right_image)
        left_image = tf.squeeze(tf.cast(left_image, tf.int32), 0)
        right_image = tf.squeeze(tf.cast(right_image, tf.int32), 0)
        return left_image, right_image

    def  cvFindCorrespondenceBM(self):
        win = self.WindowSize // 2
        shape = self.left_image.get_shape()
        disp = tf.get_variable('disp', [shape[0], shape[1], 1], tf.int32, tf.zeros_initializer)
        for i in range(0, shape[0]  - self.WindowSize + 1):
            for j in range(0, shape[1]  - self.WindowSize + 1 - self.numberOfDisparities):
                bestMatchSoFar = self.coMatch(i, j)
                indices = tf.constant([[(i + win) * self.image_height + (j + win)]])
                updates = tf.reshape(bestMatchSoFar, [1])
                disp_shape = tf.constant([self.image_height * self.image_width])
                scatter = tf.reshape(tf.scatter_nd(indices, updates, disp_shape), [shape[0], shape[1], shape[2]])
                disp = tf.add(disp, scatter)
        return disp

    def coMatch(self, i, j):
        prevdiff_1 = tf.Variable(tf.constant(2147483647))  # 32767
        prevdiff_2 = tf.Variable(tf.constant(2147483647))
        bestMatchSoFar = tf.Variable(tf.constant(self.minDisparity))
        bestMatchSoFar_1 = tf.Variable(tf.constant(self.minDisparity))
        bestMatchSoFar_2 = tf.Variable(tf.constant(self.minDisparity))
        for dispRange in range(self.minDisparity, self.numberOfDisparities):
            block_left = tf.image.crop_to_bounding_box(self.left_image, i, j,
                                                       self.WindowSize, self.WindowSize)
            block_right = tf.image.crop_to_bounding_box(self.right_image, i, j + dispRange,
                                                        self.WindowSize, self.WindowSize)
            if self.diffMethod == 'SSD':
                diff = tf.reduce_sum(tf.square(tf.subtract(block_left, block_right)))
            elif self.diffMethod == 'NCC':
                diff = tf.cast((tf.reduce_sum(block_left * block_right)) / \
                       (tf.reduce_sum(tf.square(block_left) * tf.square(block_right))), tf.float32)
                prevdiff_1 = tf.cast(prevdiff_1, tf.float32)
            else:
                diff = tf.reduce_sum(tf.abs(tf.subtract(block_left, block_right)))
            bestMatchSoFar_1 = tf.where(tf.greater(prevdiff_1, diff), dispRange,
                                        bestMatchSoFar_1)
            prevdiff_1  = tf.where(tf.greater(prevdiff_1, diff), diff,
                                   prevdiff_1)

        for dispRange in range(self.minDisparity, self.numberOfDisparities):
            co_block_right = tf.image.crop_to_bounding_box(self.right_image, i, j + bestMatchSoFar_1,
                                                           self.WindowSize, self.WindowSize)
            co_block_left = tf.image.crop_to_bounding_box(self.left_image, i, j + dispRange,
                                                          self.WindowSize, self.WindowSize)
            if self.diffMethod == 'SSD':
                diff = tf.reduce_sum(tf.square(tf.subtract(co_block_left, co_block_right)))
            elif self.diffMethod == 'NCC':
                diff = tf.cast((tf.reduce_sum(co_block_left * co_block_right)) / \
                       (tf.reduce_sum(tf.square(co_block_left) * tf.square(co_block_right))), tf.float32)
                prevdiff_2 = tf.cast(prevdiff_2, tf.float32)
            else:
                diff = tf.reduce_sum(tf.abs(tf.subtract(co_block_left, co_block_right)))
            bestMatchSoFar_2 = tf.where(tf.greater(prevdiff_2, diff), bestMatchSoFar_1 - dispRange,
                                        bestMatchSoFar_2)
            prevdiff_2 = tf.where(tf.greater(prevdiff_2, diff), diff,
                                  prevdiff_2)
            bestMatchSoFar = tf.where(tf.greater(tf.abs(bestMatchSoFar_1 - bestMatchSoFar_2), self.disp12MaxDiff),
                                      bestMatchSoFar, bestMatchSoFar_1)
        return bestMatchSoFar
