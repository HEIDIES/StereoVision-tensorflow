import tensorflow as tf
from StereoVision import BM
import numpy as np
import cv2


def run():
    left_image_path = 'test_l.png'
    right_image_path = 'test_r.png'
    left = cv2.imread(left_image_path, cv2.IMREAD_GRAYSCALE)
    right = cv2.imread(right_image_path, cv2.IMREAD_GRAYSCALE)
    left = cv2.resize(left, (32, 24))
    right = cv2.resize(right, (32, 24))
    shape = left.shape

    bm = BM(shape[0], shape[1], WindowSize=5, numberOfDisparities=4, diffMethod='SAD')
    disp = bm.cvFindCorrespondenceBM()
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    d = sess.run(disp, feed_dict={bm.left_image_raw:np.reshape(left, (bm.image_height,
                                                                      bm.image_width, 1)),
                                  bm.right_image_raw:np.reshape(right, (bm.image_height,
                                                                       bm.image_width, 1))})
    d = np.reshape(d, (24, 32))
    cv2.imshow('disparity', d)
    cv2.imshow('left_image', left)
    cv2.imshow('right_image', right)
    print(d)
    cv2.waitKey(0)


def main():
    run()


if __name__ == '__main__':
    main()
