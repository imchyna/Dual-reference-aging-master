import tensorflow as tf
from DualRefAging import DualRefAging
from os import environ
import argparse

environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


parser = argparse.ArgumentParser(description='DFA')
parser.add_argument('--is_train', type=str2bool, default=True)
parser.add_argument('--epoch', type=int, default=200, help='number of epochs')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate')
parser.add_argument('--idset', type=str, default='UTKFace', help='training id ref dataset name that stored in ./DataSet')
parser.add_argument('--ageset', type=str, default='UTKFace-input', help='training age ref dataset name that stored in ./DataSet')
parser.add_argument('--savedir', type=str, default='save', help='dir of saving checkpoints and intermediate training results')
parser.add_argument('--testdir', type=str, default='UTKFace-align-test', help='dir of testing images')
parser.add_argument('--use_trained_model', type=str2bool, default=False, help='whether train from an existing model or from scratch')
parser.add_argument('--use_init_model', type=str2bool, default=False, help='whether train from the init model if cannot find an existing model')
parser.add_argument('--chkdir', type=str, default='', help='checkpoint dir')
parser.add_argument('--chknum', type=int, default=0, help='checkpoint number')
parser.add_argument('--cof1', type=float, default=0, help='Gimg LOSS')
parser.add_argument('--cof2', type=float, default=0.001, help='G_id LOSS')
parser.add_argument('--cof3', type=float, default=0.001, help='G_age LOSS')
parser.add_argument('--cof4', type=float, default=1, help='DEX_G LOSS')
parser.add_argument('--cof5', type=float, default=1, help='DEX_id_A LOSS')

FLAGS = parser.parse_args()


def main(_):

    # print settings
    import pprint
    pprint.pprint(FLAGS)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as session:
        model = DualRefAging(
            session,  # TensorFlow session
            is_training=FLAGS.is_train,  # flag for training or testing mode
            save_dir=FLAGS.savedir,  # path to save checkpoints, samples, and summary
            idset=FLAGS.idset,  # name of the dataset in the folder ./data
            ageset=FLAGS.ageset,
            lr = FLAGS.lr
        )
        if FLAGS.is_train:
            print '\n\tTraining Mode'
            if not FLAGS.use_trained_model:
                print '\n\tTrain without trained model...'
                model.train(
                    num_epochs=FLAGS.epoch,  # number of epochs
                    use_trained_model=FLAGS.use_trained_model,
                    learning_rate= FLAGS.lr,
                    cof1=FLAGS.cof1,
                    cof2=FLAGS.cof2,
                    cof3=FLAGS.cof3,
                    cof4=FLAGS.cof4,
                    cof5=FLAGS.cof5
                )
            else:
                print '\n\tTrain with trained model...'
                model.train(
                num_epochs=FLAGS.epoch,  # number of epochs
                use_trained_model=FLAGS.use_trained_model,
                learning_rate= FLAGS.lr,
                chkdir = FLAGS.chkdir,
                chknum = FLAGS.chknum,
                cof1=FLAGS.cof1,
                cof2=FLAGS.cof2,
                cof3=FLAGS.cof3,
                cof4=FLAGS.cof4,
                cof5=FLAGS.cof5
            )
        else:
            print '\n\tTesting Mode'
            model.custom_test_new(
                testing_samples_dir=FLAGS.testdir + '/*.jpg',
                load_dir=FLAGS.chkdir,
                chknum=FLAGS.chknum
            )


if __name__ == '__main__':
    # with tf.device('/cpu:0'):
    tf.app.run()

