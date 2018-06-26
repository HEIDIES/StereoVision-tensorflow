import tensorflow as tf
from StereoVision import BM
import numpy as np
import cv2


def run():
    left_image_path = 'test_l.png'
    right_image_path = 'test_r.png'
    left = cv2.imread(left_image_path, cv2.IMREAD_GRAYSCALE)
    right = cv2.imread(right_image_path, cv2.IMREAD_GRAYSCALE)
    shape = left.shape
    bm = BM(shape[0], shape[1], diffMethod='NCC')
    disp = bm.cvFindCorrespondenceBM()
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    d = sess.run(disp, feed_dict={bm.left_image_raw:np.reshape(left, (bm.image_height,
                                                                      bm.image_width, 1)),
                                  bm.right_image_raw:np.reshape(right,(bm.image_height,
                                                                       bm.image_width, 1))})
    cv2.imshow('disparity', d)
    cv2.imshow('left_image', left)
    cv2.imshow('right_image', right)


def main():
    run()


if __name__ == '__main__':
    main()
