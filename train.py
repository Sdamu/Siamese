from dataset import *
from model import *
import logging
import time

logging.basicConfig(level=logging.DEBUG,
                    format="%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s",
                    datefmt='%b %d %H:%M')

flags = tf.flags
FLAGS = flags.FLAGS

flags.DEFINE_integer('train_iter', 500001, 'Total training iter')
flags.DEFINE_integer('validation_step', 1000, 'Total training iter')
flags.DEFINE_integer('step', 1000, 'Save after ... iteration')


with tf.name_scope("is_training"):
    is_training = tf.placeholder(dtype=tf.bool,shape=[],name='is_training')

with tf.name_scope("in"):
    left = tf.placeholder(tf.float32, [None, 72, 72, 3], name='left')
    right = tf.placeholder(tf.float32, [None, 72, 72, 3], name='right')
with tf.name_scope("similarity"):
    label = tf.placeholder(tf.int32, [None, 1], name='label')  # 1 if same, 0 if different
    label = tf.to_float(label)

left_output = SIAMESE().siamesenet(left, reuse=False, is_training=is_training)
print(left_output.shape)
# 计算右侧输入的时候使用计算左侧时相同的参数
right_output = SIAMESE().siamesenet(right, reuse=True, is_training=is_training)

# predictions, loss, accuracy = SIAMESE().contrastive_loss(left_output, right_output, label)
# model1, model2, distance, loss, acc = SIAMESE().contrastive_loss(left_output, right_output, label)

model1, model2,loss = SIAMESE().contrastive_loss(left_output, right_output, label)

global_step = tf.Variable(0, trainable=False)

# starter_learning_rate = 0.0001
# learning_rate = tf.train.exponential_decay(starter_learning_rate, global_step, 1000, 0.96, staircase=True)
# tf.summary.scalar('lr', learning_rate)
# train_step = tf.train.RMSPropOptimizer(learning_rate).minimize(loss, global_step=global_step)

# train_step = tf.train.MomentumOptimizer(0.0001, 0.99, use_nesterov=True).minimize(loss, global_step=global_step)
# train_step = tf.train.GradientDescentOptimizer(0.0001).minimize(loss, global_step=global_step)
train_step = tf.train.AdamOptimizer(0.0001).minimize(loss, global_step=global_step)



# saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=20)
    # saver.restore(sess, 'checkpoint/conv5_dataset4_color_500.ckpt')

    # setup tensorboard
    tf.summary.scalar('step', global_step)
    tf.summary.scalar('loss', loss)
    # tf.summary.scalar('accuracy', acc)
    for var in tf.trainable_variables():
        tf.summary.histogram(var.op.name, var)
    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter('train.log', sess.graph)

    left_dev_arr, right_dev_arr, similar_dev_arr = get_batch_image_array(left_dev, right_dev, similar_dev)
    is_Training = True

    ###############################

    ###############################
    # train iter
    idx = 0
    for i in range(200001):
        batch_left, batch_right, batch_similar, idx = get_batch_image_path(left_train, right_train, similar_train, idx)
        batch_left_arr, batch_right_arr, batch_similar_arr = \
            get_batch_image_array(batch_left, batch_right, batch_similar)

        time_start = time.time()
        # _, l, train_accu, summary_str = sess.run([train_step, loss,acc, merged],
        #                              feed_dict={left: batch_left_arr, right: batch_right_arr,
        #                                         label: batch_similar_arr, is_training: is_Training})
        _, l, summary_str = sess.run([train_step, loss, merged],
                                                 feed_dict={left: batch_left_arr, right: batch_right_arr,
                                                            label: batch_similar_arr, is_training: is_Training})

        # if (i + 1) % FLAGS.validation_step == 0:
        # left_dev_arr, right_dev_arr, similar_dev_arr = get_batch_image_array(left_dev, right_dev, similar_dev)

        # valid_acc = sess.run([acc],
        #                              feed_dict={left: left_dev_arr, right: right_dev_arr, label: similar_dev_arr})

        time_end = time.time()
        use_time = time_end-time_start
        writer.add_summary(summary_str, i)
        # print(valid_acc.shape)
        # print(valid_acc)
        print("\r#%d, Loss:%f, Time: %fs" % (i, l,use_time))

        # logging.info(np.average(val_distance))

        if (i) % 10 == 0 and i != 0:
            saver.save(sess, "checkpoint/conv_8_layers_model_%d.ckpt" % (i))
