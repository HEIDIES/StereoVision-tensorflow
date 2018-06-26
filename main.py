import tensorflow as tf
from StereoVision import BM
import numpy as np


def run():
    left = tf.Variable(tf.constant([[0, 0, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 0],
                                    [0, 0, 1, 2, 3, 0, 0],
                                    [0, 0, 2, 4, 6, 0, 0],
                                    [0, 0, 3, 6, 9, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 0]]), dtype=tf.int32)

    right = tf.Variable(tf.constant([[0, 0, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 0, 0, 0],
                                    [0, 0, 0, 0, 1, 2, 3],
                                    [0, 0, 0, 0, 2, 4, 6],
                                    [0, 0, 0, 0, 3, 6, 9],
                                    [0, 0, 0, 0, 0, 0, 0]]), dtype=tf.int32)
    left = tf.reshape(left, [7, 7, 1])
    right = tf.reshape(right, [7, 7, 1])
    bm = BM(7, 7, 3, numberOfDisparities=3)
    disp = bm.cvFindCorrespondenceBM()
    # dist, sad, block_left, block_right, co_block_left, co_block_right = bm.coMatch(3, 2)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    left, right = sess.run([left, right])
    """
    d, di, s, d_l, d_r, c_d_l, c_d_r = sess.run([disp, dist, sad, block_left, block_right,
                                                 co_block_left, co_block_right],
                                                feed_dict={bm.left_image:left, bm.right_image:right})
                                                """
    d = sess.run(disp, feed_dict={bm.left_image:left, bm.right_image:right})
    # print("d_l :", np.reshape(d_l, (3, 3)), "d_r :", np.reshape(d_r, (3, 3)),
          # "c_d_l :", np.reshape(c_d_l, (3, 3)), "c_d_r :", np.reshape(c_d_r, (3, 3)))
    print(np.reshape(d, (bm.image_height, bm.image_width)))


def main():
    run()


if __name__ == '__main__':
    main()