#right reserve to Yuan Zhou and Bingzhang Hu
# Date:     14th. Aug., 2018


from __future__ import division
import time
from ops import *
import datetime
from Dex import VGG_ILSVRC_16_layers
from glob import *
import tensorflow as tf
# import cv2
import numpy as np
from config import *


class DualRefAging(object):
    def __init__(self,
                 session,  # TensorFlow session
                 size_image=128,  # size the input images
                 size_age=10,  # size of the input images' ages
                 size_kernel=5,  # size of the kernels in convolution and deconvolution
                 size_batch=25,  # mini-batch size for training and testing, must be square of an integer
                 num_input_channels=3,  # number of channels of input images
                 num_encoder_channels=64,  # number of channels of the first conv layer of encoder
                 num_z_channels=50,  # number of channels of the layer z (noise or code)
                 num_categories=10,  # number of categories (age segments) in the training dataset
                 num_gen_channels=1024,  # number of channels of the first deconv layer of generator
                 enable_tile_label=True,  # enable to tile the label
                 tile_ratio=1.0,  # ratio of the length between tiled label and z
                 is_training=True,  # flag for training or testing mode
                 lr=0.0002,
                 save_dir='./save',  # path to save checkpoints, samples, and summary
                 idset='UTKFace',  # name of the dataset in the folder ./data
                 ageset='UTKFace-input'
                 ):
        self.session = session
        self.image_value_range = (-1, 1)
        self.size_image = size_image
        self.size_age = size_age
        self.size_kernel = size_kernel
        self.size_batch = size_batch
        self.num_input_channels = num_input_channels
        self.num_encoder_channels = num_encoder_channels
        self.num_z_channels = num_z_channels
        self.num_categories = num_categories
        self.num_gen_channels = num_gen_channels
        self.enable_tile_label = enable_tile_label
        self.tile_ratio = tile_ratio
        self.is_training = is_training
        self.save_dir = save_dir
        self.idset = idset
        self.ageset = ageset
        self.lr = lr
        self.conv12_coeff = 1
        self.conv22_coeff = 1
        self.conv32_coeff = 1
        self.conv42_coeff = 1
        self.conv52_coeff = 1

        # *********************************** initialization of DEX **************************
        self.DEX_conf_path = 'config.yml'  # config path
        config.read(self.DEX_conf_path)
        self.DEX_size = config.get('app').get('Rothe').get('size')
        self.DEX_means = config.get('app').get('Rothe').get('means')  # different dataset has different means

        subdir = str(self.lr) + '-' + str(datetime.datetime.now())
        # log_dir = os.path.join(os.path.expanduser(args.logs_base_dir), subdir)
        self.summary_dir = os.path.join(os.path.join(self.save_dir, 'summary'), subdir)
        self.chk_dir = os.path.join(self.save_dir, 'checkpoint', subdir)
        self.smp_dir = os.path.join(self.save_dir, 'sample', subdir)
        self.test_dir = os.path.join(self.save_dir, 'test', subdir)
        self.ref_dir = os.path.join(self.save_dir, 'ref', subdir)

        # ************************************* input to graph ********************************************************
        self.input_image_id = tf.placeholder(
            tf.float32,
            [self.size_batch, self.size_image, self.size_image, self.num_input_channels],
            name='input_ref_id'
        )  # ref id image

        self.input_image_age = tf.placeholder(
            tf.float32,
            [self.size_batch, self.size_image, self.size_image, self.num_input_channels],
            name='input_ref_age'
        )  # ref age image

        self.gender = tf.placeholder(
            tf.float32,
            [self.size_batch, 2],
            name='gender_labels'
        )
        self.z_prior = tf.placeholder(
            tf.float32,
            [self.size_batch, self.num_z_channels],
            name='z_prior'
        )
        # ************************************* build the graph *******************************************************
        print '\n\tBuilding graph ...'
        # encoder_I: input ref id image --> z_id_I
        self.z_id_I = self.encoder_I(
            image=self.input_image_id,
            norm=True #convert id_ref to [-1,1]
        )

        self.z_age_I = self.encoder_I(
            image=self.input_image_age,
            norm=True, #convert age_ref to [-1,1]
            reuse=True
        )

        # encoder_A: input ref age image -->z_age_A
        self.z_age_A, self.realage_conv12, self.realage_conv22, self.realage_conv32, \
        self.realage_conv42, self.realage_conv52 = self.encoder_A(
            images=self.input_image_age,
            is_training=False
        )

        self.z_id_A, self.realid_A_conv12, self.realid_A_conv22, self.realid_A_conv32,\
        self.realid_A_conv42, self.realid_A_conv52 = self.encoder_A(
            images=self.input_image_id,
            reuse=True,
            is_training=False
        )

        # self.A_age = self.proc_age(self.z_age_A)
        # generator: z + label --> generated image ori_G in [0,255], G in [-1,1]
        self.ori_G, self.G, self.age_age = self.generator(
            z_id=self.z_id_I,
            z_age=self.z_age_A,
            enable_tile_label=self.enable_tile_label,
            tile_ratio=self.tile_ratio,
            reuse=False
        )

        # self.I_age = self.proc_age(self.z_id_A)
        # generate image with id_id and id_age, to compare with id_ref and get the EG_loss
        self.ori_id_G, self.id_G, self.id_age = self.generator(
            z_id=self.z_id_I,
            z_age=self.z_id_A,
            enable_tile_label=self.enable_tile_label,
            tile_ratio=self.tile_ratio,
            reuse=True
        )

        self.z_G_A, self.fakeage_conv12, self.fakeage_conv22, self.fakeage_conv32,\
        self.fakeage_conv42, self.fakeage_conv52 = self.encoder_A(
            images=self.ori_G,
            reuse=True,
            is_training=False
        )

        self.z_id_G_A, self.fakeid_A_conv12, self.fakeid_A_conv22, self.fakeid_A_conv32,\
        self.fakeid_A_conv42, self.fakeid_A_conv52 = self.encoder_A(
            images=self.ori_id_G,
            reuse=True,
            is_training=False
        )

        self.z_G_I = self.encoder_I(
            image=self.G,
            norm=False,
            reuse=True
        )

        # discriminator on z
        self.D_z, self.Dz_logits = self.discriminator_z_I(
            z=self.z_id_I,
            is_training=self.is_training
        )

        # discriminator on z_prior
        self.Dz_prior, self.Dz_prior_logits = self.discriminator_z_I(
            z=self.z_prior,
            reuse=True,
            is_training=self.is_training
        )

        # discriminator on G
        self.Dimg_G, self.Dimg_G_logits = self.discriminator_img(
            image=self.G,
            norm=False,
            is_training=self.is_training
        )

        self.Dimg_id, self.Dimg_id_logits = self.discriminator_img(
            image=self.input_image_id,
            norm=True,
            is_training=self.is_training,
            reuse=True
        )

        self.Dimg_age, self.Dimg_age_logits = self.discriminator_img(
            image=self.input_image_age,
            norm=True,
            is_training=self.is_training,
            reuse=True
        )

        self.Did_id, self.Did_id_logits = self.discriminator_id(
            # id_ref and id_ref are the same identity
            z_id1=self.z_id_I,
            z_id2=self.z_id_I,
            is_training=self.is_training
        )

        self.Did_age, self.Did_age_logits = self.discriminator_id(
            # id_ref and age_ref are the different identity
            z_id1=self.z_id_I,
            z_id2=self.z_age_I,
            reuse=True,
            is_training=self.is_training
        )

        self.Did_G, self.Did_G_logits = self.discriminator_id(
            # id_ref and G are the same identity
            z_id1=self.z_id_I,
            z_id2=self.z_G_I,
            reuse=True,
            is_training=self.is_training
        )

        # discriminator on age_prior
        self.Dage_age, self.Dage_age_logits = self.discriminator_age(
            z_age1=self.z_age_A,
            z_age2=self.z_age_A,
            is_training=self.is_training
        )

        self.Dage_id, self.Dage_id_logits = self.discriminator_age(
            z_age1=self.z_age_A,
            z_age2=self.z_id_A,
            reuse=True,
            is_training=self.is_training
        )

        self.Dage_G, self.Dage_G_logits = self.discriminator_age(
            z_age1=self.z_age_A,
            z_age2=self.z_G_A,
            reuse=True,
            is_training=self.is_training
        )

        # ************************************* loss functions *******************************************************
        ### summary of DEX layer losses
        self.age_conv12_loss = tf.reduce_mean(tf.abs(self.realage_conv12 - self.fakeage_conv12)) / 224. / 224.
        self.age_conv22_loss = tf.reduce_mean(tf.abs(self.realage_conv22 - self.fakeage_conv22)) / 112. / 112.
        self.age_conv32_loss = tf.reduce_mean(tf.abs(self.realage_conv32 - self.fakeage_conv32)) / 56. / 56.
        self.age_conv42_loss = tf.reduce_mean(tf.abs(self.realage_conv42 - self.fakeage_conv42)) / 28. / 28.
        self.age_conv52_loss = tf.reduce_mean(tf.abs(self.realage_conv52 - self.fakeage_conv52)) / 14. / 14.
        self.DEX_G_loss = self.conv12_coeff * self.age_conv12_loss + self.conv22_coeff * self.age_conv22_loss + \
                          self.conv32_coeff * self.age_conv32_loss + self.conv42_coeff * self.age_conv42_loss + self.conv52_coeff * self.age_conv52_loss
        self.id_A_conv12_loss = tf.reduce_mean(tf.abs(self.realid_A_conv12 - self.fakeid_A_conv12)) / 224. / 224.
        self.id_A_conv22_loss = tf.reduce_mean(tf.abs(self.realid_A_conv22 - self.fakeid_A_conv22)) / 112. / 112.
        self.id_A_conv32_loss = tf.reduce_mean(tf.abs(self.realid_A_conv32 - self.fakeid_A_conv32)) / 56. / 56.
        self.id_A_conv42_loss = tf.reduce_mean(tf.abs(self.realid_A_conv42 - self.fakeid_A_conv42)) / 28. / 28.
        self.id_A_conv52_loss = tf.reduce_mean(tf.abs(self.realid_A_conv52 - self.fakeid_A_conv52)) / 14. / 14.
        self.DEX_id_A_loss = self.conv12_coeff * self.id_A_conv12_loss + self.conv22_coeff * self.id_A_conv22_loss + \
                             self.conv32_coeff * self.id_A_conv32_loss + self.conv42_coeff * self.id_A_conv42_loss + self.conv52_coeff * self.id_A_conv52_loss

        image = self.input_image_id * (self.image_value_range[-1] - self.image_value_range[0]) / 255.0 + \
                self.image_value_range[0]  # format inputing in [-1,1]
        self.EG_loss = tf.reduce_mean(tf.abs(image - self.id_G))  # L1 loss

        # loss function of discriminator on z
        self.Dz_prior_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.Dz_prior_logits,
                                                    labels=tf.ones_like(self.Dz_prior_logits))
        )
        self.Dz_z_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.Dz_logits,
                                                    labels=tf.zeros_like(self.Dz_logits))
        )
        self.Ez_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.Dz_logits,
                                                    labels=tf.ones_like(self.Dz_logits))
        )
        # loss function of discriminator on image

        self.Dimg_id_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.Dimg_id_logits,
                                                    labels=tf.ones_like(self.Dimg_id_logits))
        )
        self.Dimg_age_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.Dimg_age_logits,
                                                    labels=tf.ones_like(self.Dimg_age_logits))
        )
        self.Dimg_G_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.Dimg_G_logits,
                                                    labels=tf.zeros_like(self.Dimg_G_logits))
        )
        self.Gimg_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.Dimg_G_logits,
                                                    labels=tf.ones_like(self.Dimg_G_logits))
        )
        # loss function of discriminator on ID
        self.Did_id_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.Did_id_logits,
                                                    labels=tf.ones_like(self.Did_id_logits))
        )

        self.Did_age_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.Did_age_logits,
                                                    labels=tf.zeros_like(self.Did_age_logits))
        )

        self.Did_G_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.Did_G_logits,
                                                    labels=tf.zeros_like(self.Did_G_logits))
        )

        self.Gid_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.Did_G_logits,
                                                    labels=tf.ones_like(self.Did_G_logits))
        )
        # loss function of discriminator on AGE
        self.Dage_age_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.Dage_age_logits,
                                                    labels=tf.ones_like(self.Dage_age_logits))
        )

        self.Dage_id_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.Dage_id_logits,
                                                    labels=tf.zeros_like(self.Dage_id_logits))
        )

        self.Dage_G_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.Dage_G_logits,
                                                    labels=tf.zeros_like(self.Dage_G_logits))
        )

        self.Gage_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.Dage_G_logits,
                                                    labels=tf.ones_like(self.Dage_G_logits))
        )

        # total variation to smooth the generated image
        tv_y_size = self.size_image
        tv_x_size = self.size_image
        self.tv_loss = (
                           (tf.nn.l2_loss(
                               self.G[:, 1:, :, :] - self.G[:, :self.size_image - 1, :, :]) / tv_y_size) +
                           (tf.nn.l2_loss(self.G[:, :, 1:, :] - self.G[:, :, :self.size_image - 1,
                                                                :]) / tv_x_size)) / self.size_batch

        # *********************************** trainable variables ****************************************************
        trainable_variables = tf.trainable_variables()
        # variables of encoder
        self.E_variables = [var for var in trainable_variables if 'E_' in var.name]
        # variables of generator
        self.G_variables = [var for var in trainable_variables if 'G_' in var.name]
        # variables of discriminator on z
        self.Dz_variables = [var for var in trainable_variables if 'Dz_' in var.name]
        # variables of discriminator on image
        self.Dimg_variables = [var for var in trainable_variables if 'Dimg_' in var.name]
        self.Did_variables = [var for var in trainable_variables if 'Did_' in var.name]
        self.Dage_variables = [var for var in trainable_variables if 'Dage_' in var.name]

        # ************************************* collect the summary ***************************************
        self.z_summary = tf.summary.histogram('z_id_I', self.z_id_I)
        self.z_prior_summary = tf.summary.histogram('z_prior', self.z_prior)
        self.DEX_G_loss_summary = tf.summary.scalar('DEX_G_loss', self.DEX_G_loss)
        self.DEX_id_A_loss_summary = tf.summary.scalar('DEX_id_A_loss', self.DEX_id_A_loss)
        self.EG_loss_summary = tf.summary.scalar('EG_loss', self.EG_loss)
        self.Dz_z_loss_summary = tf.summary.scalar('Dz_z_loss', self.Dz_z_loss)
        self.Dz_prior_loss_summary = tf.summary.scalar('Dz_prior_loss', self.Dz_prior_loss)
        self.Ez_loss_summary = tf.summary.scalar('Ez_loss', self.Ez_loss)
        self.Dz_logits_summary = tf.summary.histogram('Dz_logits', self.Dz_logits)
        self.Dz_prior_logits_summary = tf.summary.histogram('Dz_prior_logits', self.Dz_prior_logits)
        self.Dimg_id_loss_summary = tf.summary.scalar('Dimg_id_loss', self.Dimg_id_loss)
        self.Dimg_age_loss_summary = tf.summary.scalar('Dimg_age_loss', self.Dimg_age_loss)
        self.Dimg_G_loss_summary = tf.summary.scalar('Dimg_G_loss', self.Dimg_G_loss)
        self.Gimg_loss_summary = tf.summary.scalar('Gimg_loss', self.Gimg_loss)
        self.Dimg_id_logits_summary = tf.summary.histogram('Dimg_id_logits', self.Dimg_id_logits)
        self.Dimg_age_logits_summary = tf.summary.histogram('Dimg_age_logits', self.Dimg_age_logits)
        self.Dimg_G_logits_summary = tf.summary.histogram('Dimg_G_logits', self.Dimg_G_logits)

        self.Did_id_loss_summary = tf.summary.scalar('Did_id_loss', self.Did_id_loss)
        self.Did_age_loss_summary = tf.summary.scalar('Did_age_loss', self.Did_age_loss)
        self.Did_G_loss_summary = tf.summary.scalar('Did_G_loss', self.Did_G_loss)
        self.Gid_loss_summary = tf.summary.scalar('Gid_loss', self.Gid_loss)
        self.Did_G_logits_summary = tf.summary.histogram('Did_G_logits', self.Did_G_logits)
        self.Did_id_logits_summary = tf.summary.histogram('Did_id_logits', self.Did_id_logits)
        self.Did_age_logits_summary = tf.summary.histogram('Did_age_logits', self.Did_age_logits)

        self.Dage_age_loss_summary = tf.summary.scalar('Dage_age_loss', self.Dage_age_loss)
        self.Dage_id_loss_summary = tf.summary.scalar('Dage_id_loss', self.Dage_id_loss)
        self.Dage_G_loss_summary = tf.summary.scalar('Dage_G_loss', self.Dage_G_loss)
        self.Gage_loss_summary = tf.summary.scalar('Gage_loss', self.Gage_loss)
        self.Dage_age_logits_summary = tf.summary.histogram('Dage_age_logits', self.Dage_age_logits)
        self.Dage_id_logits_summary = tf.summary.histogram('Dage_id_logits', self.Dage_id_logits)
        self.Dage_G_logits_summary = tf.summary.histogram('Dage_G_logits', self.Dage_G_logits)

        self.img_G_summary = tf.summary.image('image_G', self.ori_G, max_outputs=3)
        self.img_id_summary = tf.summary.image('image_id_G', self.ori_id_G, max_outputs=3)
        # for saving the graph and variables
        self.saver = tf.train.Saver(max_to_keep=0)

    def train(self,
              num_epochs=200,  # number of epochs
              learning_rate=0.0002,  # learning rate of optimizer
              beta1=0.5,  # parameter for Adam optimizer
              decay_rate=1.0,  # learning rate decay (0, 1], 1 means no decay
              enable_shuffle=True,  # enable shuffle of the dataset
              use_trained_model=True,  # use the saved checkpoint to initialize the network
              cof1=0,  # the weights of adversarial loss and TV loss
              cof2=0,
              cof3=0,
              cof4=0,
              cof5=0,
              cof6=0,
              chkdir='',
              chknum=0
              ):
        print(learning_rate)
        # *************************** load sample file names ******************************************************
        train_list = np.load('UTKnew-4.npy')
        size_data = len(train_list)
        np.random.seed(seed=2018)  # decrease the training number
        if enable_shuffle:
            np.random.shuffle(train_list)
        sample_files = train_list[0:self.size_batch]
        train_list = train_list[self.size_batch + 1:size_data]
        id_tmp = sample_files[:, 0]
        age_tmp = sample_files[:, 1]
#load and save sample_id_ref
        id_path = os.path.join(self.ref_dir, 'id')
        if not os.path.exists(id_path):
            os.makedirs(id_path)
        batch_id = [load_image(
            image_path=id_file,
            image_size=self.size_image,
            image_value_range=self.image_value_range,
            is_gray=(self.num_input_channels == 1),
            save=True,
            save_path=id_path
        ) for id_file in id_tmp]
        sample_id_ref = np.array(batch_id).astype(np.float32)
#load and save sample_age_ref
        age_path = os.path.join(self.ref_dir, 'age')
        if not os.path.exists(age_path):
            os.makedirs(age_path)
        batch_age = [load_image(
            image_path=age_file,
            image_size=self.size_image,
            image_value_range=self.image_value_range,
            is_gray=(self.num_input_channels == 1),
            save=True,
            save_path=age_path
        ) for age_file in age_tmp]
        sample_age_ref = np.array(batch_age).astype(np.float32)
        np.save(self.ref_dir, sample_files)  # save ref files info

        with tf.variable_scope('opt'):
            # *********************************** optimizer **************************************************************
            self.loss_E = self.Ez_loss # + cof1 * self.Ez_loss + cof2 * self.Gimg_loss + cof3 * self.Gid_loss + cof4 * self.Gage_loss + cof5 * self.DEX_G_loss + cof6 * self.DEX_id_A_loss + 0.001 * self.tv_loss  # self.EG_loss+
            self.loss_G = self.EG_loss + cof1 * self.Gimg_loss + cof2 * self.Gid_loss + cof3 * self.Gage_loss + cof4 * self.DEX_G_loss + cof5 * self.DEX_id_A_loss + 0.001 * self.tv_loss  # slightly increase the params
            self.loss_Dz = self.Dz_prior_loss + self.Dz_z_loss
            self.loss_Dimg = self.Dimg_id_loss + self.Dimg_age_loss + self.Dimg_G_loss
            self.loss_Did = self.Did_id_loss + self.Did_age_loss + self.Did_G_loss
            self.loss_Dage = self.Dage_age_loss + self.Dage_id_loss + self.Dage_G_loss

            # set learning rate decay
            if chknum > 0:
                self.EG_global_step = tf.Variable(chknum, trainable=False, name='global_step')
            else:
                self.EG_global_step = tf.Variable(0, trainable=False, name='global_step')
            EG_learning_rate = tf.train.exponential_decay(
                learning_rate=learning_rate,
                global_step=self.EG_global_step,
                decay_steps=size_data / self.size_batch * 2,
                decay_rate=decay_rate,
                staircase=True
            )
            # optimizer for encoder_I
            self.E_optimizer = tf.train.AdamOptimizer(
                beta1=beta1,
                learning_rate=EG_learning_rate
            ).minimize(
                loss=self.loss_E,
                global_step=self.EG_global_step,
                var_list=self.E_variables
            )

            # optimizer for G
            self.G_optimizer = tf.train.AdamOptimizer(
                beta1=beta1,
                learning_rate=EG_learning_rate
            ).minimize(
                loss=self.loss_G,
                global_step=self.EG_global_step,
                var_list=self.G_variables
            )

            # optimizer for D_z
            self.Dz_optimizer = tf.train.AdamOptimizer(
                beta1=beta1,
                learning_rate=EG_learning_rate
            ).minimize(
                loss=self.loss_Dz,
                var_list=self.Dz_variables
            )

            # optimizer for D_img
            self.Dimg_optimizer = tf.train.AdamOptimizer(
                beta1=beta1,
                learning_rate=EG_learning_rate
            ).minimize(
                loss=self.loss_Dimg,
                var_list=self.Dimg_variables
            )

            # optimizer for D_id
            self.Did_optimizer = tf.train.AdamOptimizer(
                beta1=beta1,
                learning_rate=EG_learning_rate
            ).minimize(
                loss=self.loss_Did,
                var_list=self.Did_variables
            )

            # optimizer for D_age
            self.Dage_optimizer = tf.train.AdamOptimizer(
                beta1=beta1,
                learning_rate=EG_learning_rate
            ).minimize(
                loss=self.loss_Dage,
                var_list=self.Dage_variables
            )

        # *********************************** tensorboard *************************************************************
        # for visualizatiooptimizern (TensorBoard): $ tensorboard --logdir path/to/log-directory
        self.EG_learning_rate_summary = tf.summary.scalar('EG_learning_rate', EG_learning_rate)
        self.summary = tf.summary.merge([
            self.z_summary, self.z_prior_summary,
            self.Dz_z_loss_summary, self.Dz_prior_loss_summary,
            self.DEX_G_loss_summary, self.DEX_id_A_loss_summary,
            self.EG_loss_summary, self.Ez_loss_summary,
            self.Dimg_id_loss_summary, self.Dimg_G_loss_summary,
            self.Gimg_loss_summary, self.img_G_summary, self.img_id_summary,
            self.Dimg_id_logits_summary, self.Dimg_age_logits_summary, self.Dimg_G_logits_summary,
            self.Did_id_loss_summary, self.Did_age_loss_summary, self.Did_G_loss_summary,
            self.Gid_loss_summary,
            self.Did_id_logits_summary, self.Did_age_logits_summary, self.Did_G_logits_summary,
            self.Dage_age_loss_summary, self.Dage_id_loss_summary, self.Dage_G_loss_summary,
            self.Gage_loss_summary,
            self.Dage_age_logits_summary, self.Dage_id_logits_summary, self.Dage_G_logits_summary,
            self.EG_learning_rate_summary
        ])
        self.writer = tf.summary.FileWriter(self.summary_dir, self.session.graph)

        # ******************************************* training *******************************************************
        # initialize the graph
        tf.global_variables_initializer().run()
        # ****************** load weights to Dex *******************
        self.Dex.load('Dex.npy', self.session)

        # load check point
        if use_trained_model:
            if self.load_checkpoint(chkdir, chknum):
                print("\tSUCCESS ^_^")
            else:
                print("\tFAILED >_<!")

        print '\n\tBegin training ...'
        # epoch iteration
        num_batches = len(train_list) // self.size_batch
        for epoch in range(num_epochs):
            if enable_shuffle:
                np.random.shuffle(train_list)
            for ind_batch in range(num_batches):
                start_time = time.time()
                # read batch id ref images and labels
                batch_files = train_list[ind_batch * self.size_batch:(ind_batch + 1) * self.size_batch]
                batch_id = batch_files[:, 0]
                batch_age = batch_files[:, 1]
                batch = [load_image(
                    image_path=id_file,
                    image_size=self.size_image,
                    image_value_range=self.image_value_range,
                    is_gray=(self.num_input_channels == 1),
                ) for id_file in batch_id]
                if self.num_input_channels == 1:
                    id_ref = np.array(batch).astype(np.float32)[:, :, :, None]
                else:
                    id_ref = np.array(batch).astype(np.float32)
                batch = [load_image(
                    image_path=age_file,
                    image_size=self.size_image,
                    image_value_range=self.image_value_range,
                    is_gray=(self.num_input_channels == 1),
                ) for age_file in batch_age]
                if self.num_input_channels == 1:
                    age_ref = np.array(batch).astype(np.float32)[:, :, :, None]
                else:
                    age_ref = np.array(batch).astype(np.float32)

                # prior distribution on the prior of z
                batch_z_prior = np.random.uniform(
                    self.image_value_range[0],
                    self.image_value_range[-1],
                    [self.size_batch, self.num_z_channels]
                ).astype(np.float32)


                # update
                # age_age,id_age,ori_G, ori_id_G, G,id_G,\
                _, _, _, _, _, _, Dz_p_err, Dz_z_err, Ez_err, Dimg_id_err, Dimg_age_err, Dimg_G_err, Gimg_err, EG_err, \
                Did_id_err, Did_age_err, Did_G_err, Gid_err, DEX_G_err, DEX_id_A_err, \
                Dage_age_err, Dage_id_err, Dage_G_err, Gage_err, TV = self.session.run(
                    fetches=[
                        # self.age_age, self.id_age, self.ori_G, self.ori_id_G, self.G, self.id_G,
                        self.Dz_optimizer,
                        self.Dimg_optimizer,
                        self.Did_optimizer,
                        self.Dage_optimizer,
                        self.E_optimizer,
                        self.G_optimizer,
                        self.Dz_prior_loss,
                        self.Dz_z_loss,
                        self.Ez_loss,
                        self.Dimg_id_loss,
                        self.Dimg_age_loss,
                        self.Dimg_G_loss,
                        self.Gimg_loss,
                        self.EG_loss,
                        self.Did_id_loss,
                        self.Did_age_loss,
                        self.Did_G_loss,
                        self.Gid_loss,
                        self.DEX_G_loss,
                        self.DEX_id_A_loss,
                        self.Dage_age_loss,
                        self.Dage_id_loss,
                        self.Dage_G_loss,
                        self.Gage_loss,
                        self.tv_loss
                    ],
                    feed_dict={
                        self.input_image_id: id_ref,
                        self.input_image_age: age_ref,
                        self.z_prior: batch_z_prior
                    }
                )

                print("\nEpoch: [%3d/%3d] Batch: [%3d/%3d]\n\tEG_err=%.4f\tTV=%.4f" %
                      (epoch + 1, num_epochs, ind_batch + 1, num_batches, EG_err, TV))
                print("\tEz=%.4f\tDz_p=%.4f\tDz_z=%.4f" % (Ez_err, Dz_p_err, Dz_z_err))
                print("\tG_Dimg=%.4f\tDimg_id=%.4f\tDimg_G=%.4f" % (
                    Gimg_err, Dimg_id_err, Dimg_G_err))
                print("\tG_Did=%.4f\tDid_id=%.4f\tDid_G=%.4f" % (Gid_err, Did_id_err, Did_G_err))
                print("\tG_Dage=%.4f\tDage_age=%.4f\tDage_G=%.4f" % (Gage_err, Dage_age_err, Dage_G_err))

                # estimate left run time
                elapse = time.time() - start_time
                time_left = ((num_epochs - epoch - 1) * num_batches + (num_batches - ind_batch - 1)) * elapse
                print("\tTime left: %02d:%02d:%02d" %
                      (int(time_left / 3600), int(time_left % 3600 / 60), time_left % 60))

                # add to summary
                summary = self.summary.eval(
                    feed_dict={
                        self.input_image_id: id_ref,
                        self.input_image_age: age_ref,
                        self.z_prior: batch_z_prior
                    }
                )
                self.writer.add_summary(summary, self.EG_global_step.eval())

                if (np.mod(ind_batch, 500) == 499):#and (epoch >= ):
                    size_frame = int(np.sqrt(self.size_batch))
                    name = str(epoch) + '-' + str(ind_batch) + '.png'
                    self.sample(id_ref=sample_id_ref, age_ref=sample_id_ref, name=name)
                    self.test(id_ref=sample_id_ref, age_ref=sample_age_ref, name=name, size_frame=size_frame)
                    self.save_checkpoint()

            # save sample images for each epoch
            name = '{:02d}.png'.format(epoch + 1)
            self.sample(id_ref=sample_id_ref, age_ref=sample_id_ref, name=name)  # save sample file

            # *******************save test files **************************************
            size_frame = int(np.sqrt(self.size_batch))
            if epoch == 0:
                if not os.path.exists(self.test_dir):
                    os.makedirs(self.test_dir)
                save_batch_images(
                    batch_images=sample_id_ref,
                    save_path=os.path.join(self.test_dir, 'input_id.png'),
                    image_value_range=self.image_value_range,
                    size_frame=[size_frame, size_frame]
                )
                save_batch_images(
                    batch_images=sample_age_ref,
                    save_path=os.path.join(self.test_dir, 'input_age.png'),
                    image_value_range=self.image_value_range,
                    size_frame=[size_frame, size_frame]
                )
            self.test(id_ref=sample_id_ref, age_ref=sample_age_ref, name=name, size_frame=size_frame)

            # save checkpoint for each 5 epoch
            # if np.mod(epoch, 5) == 4:
            self.save_checkpoint()

        # save the trained model
        self.save_checkpoint()
        # close the summary writer
        self.writer.close()

    def encoder_I(self, image, norm=True, reuse=False):
        with tf.variable_scope('encoder_I'):
            input_img = image
            if norm: #convert image from [0,255] to [-1,1]
                input_img = input_img * (self.image_value_range[-1] - self.image_value_range[0]) / 255.0 + \
                            self.image_value_range[0]  # format inputing in [-1,1]
            num_layers = int(np.log2(self.size_image)) - int(self.size_kernel / 2)
            current = input_img
            # conv layers with stride 2
            for i in range(num_layers):
                name = 'E_conv' + str(i)
                current = conv2d(
                    input_map=current,
                    num_output_channels=self.num_encoder_channels * (2 ** i),
                    size_kernel=self.size_kernel,
                    name=name,
                    reuse=reuse
                )
                current = tf.nn.relu(current)

            # fully connection layer
            name = 'E_fc'
            current = fc(
                input_vector=tf.reshape(current, [self.size_batch, -1]),
                num_output_length=self.num_z_channels,
                name=name,
                reuse=reuse
            )

            # output
            return tf.nn.tanh(current)

    def encoder_A(self, images, reuse=False, is_training=False):
        data_Dex = tf.image.resize_images(images, self.DEX_size)  # resize the inputting to 224*224*3 for Dex
        data_Dex = tf.subtract(data_Dex, self.DEX_means)  # format inputting by substract with means
        self.Dex = VGG_ILSVRC_16_layers({'data': data_Dex}, reuse=reuse, trainable=is_training)
        ip2 = self.Dex.layers['prob']
        conv12 = self.Dex.layers['conv1_2']
        conv22 = self.Dex.layers['conv2_2']
        conv32 = self.Dex.layers['conv3_2']
        conv42 = self.Dex.layers['conv4_2']
        conv52 = self.Dex.layers['conv5_2']
        pred = tf.nn.softmax(ip2)
        return pred, conv12, conv22, conv32, conv42, conv52

    def generator(self, z_id, z_age, reuse=False, enable_tile_label=True, tile_ratio=1.0):
        with tf.variable_scope('generator'):
            num_layers = int(np.log2(self.size_image)) - int(self.size_kernel / 2)
            if enable_tile_label:
                duplicate = int(self.num_z_channels * tile_ratio / self.num_categories)
            else:
                duplicate = 1

            age = self.proc_age(z_age)  # transform z_age according to the categories
            age = tf.cast(age,dtype=tf.float32)
            # age = z_age
            z = concat_label(z_id, age, duplicate=duplicate)

            size_mini_map = int(self.size_image / 2 ** num_layers)
            # fc layer
            name = 'G_fc'
            current = fc(
                input_vector=z,
                num_output_length=self.num_gen_channels * size_mini_map * size_mini_map,
                name=name,
                reuse=reuse
            )
            # reshape to cube for deconv
            current = tf.reshape(current, [-1, size_mini_map, size_mini_map, self.num_gen_channels])
            current = tf.nn.relu(current)
            current = concat_label(current, age)
            # deconv layers with stride 2
            for i in range(num_layers):
                name = 'G_deconv' + str(i)
                current = deconv2d(
                    input_map=current,
                    output_shape=[self.size_batch,
                                  size_mini_map * 2 ** (i + 1),
                                  size_mini_map * 2 ** (i + 1),
                                  int(self.num_gen_channels / 2 ** (i + 1))],
                    size_kernel=self.size_kernel,
                    name=name,
                    reuse=reuse
                )
                current = tf.nn.relu(current)
                current = concat_label(current, age)
            name = 'G_deconv' + str(i + 1)
            current = deconv2d(
                input_map=current,
                output_shape=[self.size_batch,
                              self.size_image,
                              self.size_image,
                              int(self.num_gen_channels / 2 ** (i + 2))],
                size_kernel=self.size_kernel,
                stride=1,
                name=name,
                reuse=reuse
            )
            current = tf.nn.relu(current)
            current = concat_label(current, age)
            name = 'G_deconv' + str(i + 2)
            current = deconv2d(
                input_map=current,
                output_shape=[self.size_batch,
                              self.size_image,
                              self.size_image,
                              self.num_input_channels],
                size_kernel=self.size_kernel,
                stride=1,
                name=name,
                reuse=reuse
            )

            # output
            output = (tf.nn.tanh(current) + 1) * 127.5
            return output, tf.nn.tanh(current), age

    def discriminator_z_I(self, z, is_training=True, reuse=False, num_hidden_layer_channels=(64, 32, 16),
                          enable_bn=True):
        with tf.variable_scope('discriminator_z'):

            current = z
            # fully connection layer
            for i in range(len(num_hidden_layer_channels)):
                name = 'Dz_fc' + str(i)
                current = fc(
                    input_vector=current,
                    num_output_length=num_hidden_layer_channels[i],
                    name=name,
                    reuse=reuse
                )
                current = tf.nn.relu(current)
                if enable_bn:
                    name = 'Dz_bn' + str(i)
                    current = tf.contrib.layers.batch_norm(
                        current,
                        scale=False,
                        is_training=is_training,
                        scope=name,
                        reuse=reuse
                    )

            # output layer
            name = 'Dz_fc' + str(i + 1)
            current = fc(
                input_vector=current,
                num_output_length=1,
                name=name,
                reuse=reuse
            )
            return tf.nn.sigmoid(current), current

    def discriminator_img(self, image, norm = False, is_training=True, reuse=False,
                          num_hidden_layer_channels=(16, 32, 64, 128), enable_bn=True):
        if norm:
            image = image * (self.image_value_range[-1] - self.image_value_range[0]) / 255.0 + \
                        self.image_value_range[0]  # format inputing in [-1,1]
        with tf.variable_scope('discriminator_img'):
            num_layers = len(num_hidden_layer_channels)
            current = image
            # current = concat_label(current, age)
            # conv layers with stride 2
            for i in range(num_layers):
                name = 'Dimg_conv' + str(i)
                current = conv2d(
                    input_map=current,
                    num_output_channels=num_hidden_layer_channels[i],
                    size_kernel=self.size_kernel,
                    name=name,
                    reuse=reuse
                )
                current = tf.nn.relu(current)
                if enable_bn:
                    name = 'Dimg_bn' + str(i)
                    current = tf.contrib.layers.batch_norm(
                        current,
                        scale=False,
                        is_training=is_training,
                        scope=name,
                        reuse=reuse
                    )
                # current = concat_label(current, age)

            # fully connection layer
            name = 'Dimg_fc1'
            current = fc(
                input_vector=tf.reshape(current, [self.size_batch, -1]),
                num_output_length=1024,
                name=name,
                reuse=reuse
            )
            current = lrelu(current)
            name = 'Dimg_fc1_bn'
            current = tf.contrib.layers.batch_norm(
                current,
                scale=False,
                is_training=is_training,
                scope=name,
                reuse=reuse
            )
            # current = concat_label(current, age)

            name = 'Dimg_fc2'
            current = fc(
                input_vector=current,
                num_output_length=1,
                name=name,
                reuse=reuse
            )
            # output
            return tf.nn.sigmoid(current), current

    def discriminator_id(self, z_id1, z_id2, is_training=True, reuse=False, enable_bn=True):
        with tf.variable_scope('discriminator_id'):
            ## cosine distance
            # current = siamese_cosine_loss(id1, id2)
            current = tf.einsum('ij,ik->ijk', z_id1, z_id2)
            name = 'Did_fc1'
            current = fc(
                input_vector=tf.reshape(current, [self.size_batch, -1]),
                num_output_length=1024,
                name=name,
                reuse=reuse
            )
            current = lrelu(current)
            if enable_bn:
                name = 'Did_bn1'
                current = tf.contrib.layers.batch_norm(
                    current,
                    scale=False,
                    is_training=is_training,
                    scope=name,
                    reuse=reuse
                )
            name = 'Did_fc2'
            current = fc(
                input_vector=current,
                num_output_length=512,
                name=name,
                reuse=reuse
            )
            current = lrelu(current)
            if enable_bn:
                name = 'Did_bn2'
                current = tf.contrib.layers.batch_norm(
                    current,
                    scale=False,
                    is_training=is_training,
                    scope=name,
                    reuse=reuse
                )
            name = 'Did_fc3'
            current = fc(
                input_vector=current,
                num_output_length=256,
                name=name,
                reuse=reuse
            )
            current = lrelu(current)
            if enable_bn:
                name = 'Did_bn3'
                current = tf.contrib.layers.batch_norm(
                    current,
                    scale=False,
                    is_training=is_training,
                    scope=name,
                    reuse=reuse
                )
            name = 'Did_fc4'
            current = fc(
                input_vector=current,
                num_output_length=128,
                name=name,
                reuse=reuse
            )
            current = lrelu(current)
            if enable_bn:
                name = 'Did_bn4'
                current = tf.contrib.layers.batch_norm(
                    current,
                    scale=False,
                    is_training=is_training,
                    scope=name,
                    reuse=reuse
                )
            name = 'Did_fc5'
            current = fc(
                input_vector=current,
                num_output_length=64,
                name=name,
                reuse=reuse
            )
            current = lrelu(current)
            if enable_bn:
                name = 'Did_bn5'
                current = tf.contrib.layers.batch_norm(
                    current,
                    scale=False,
                    is_training=is_training,
                    scope=name,
                    reuse=reuse
                )
            name = 'Did_fc6'
            current = fc(
                input_vector=current,
                num_output_length=32,
                name=name,
                reuse=reuse
            )
            current = lrelu(current)
            if enable_bn:
                name = 'Did_bn6'
                current = tf.contrib.layers.batch_norm(
                    current,
                    scale=False,
                    is_training=is_training,
                    scope=name,
                    reuse=reuse
                )
            name = 'Did_fc7'
            current = fc(
                input_vector=current,
                num_output_length=16,
                name=name,
                reuse=reuse
            )
            current = lrelu(current)
            if enable_bn:
                name = 'Did_bn7'
                current = tf.contrib.layers.batch_norm(
                    current,
                    scale=False,
                    is_training=is_training,
                    scope=name,
                    reuse=reuse
                )
            name = 'Did_fc8'
            current = fc(
                input_vector=current,
                num_output_length=1,
                name=name,
                reuse=reuse
            )
            # output
            return tf.nn.sigmoid(current), current

    def discriminator_age(self, z_age1, z_age2, is_training=True, reuse=False, enable_bn=True):
        with tf.variable_scope('discriminator_age'):
            current = tf.concat((z_age1, z_age2), axis=1)
            name = 'Dage_fc1'
            current = fc(
                input_vector=current,  # tf.reshape(current, [self.size_batch, -1]),
                num_output_length=404,
                name=name,
                reuse=reuse
            )
            current = lrelu(current)
            if enable_bn:
                name = 'Dage_bn1'
                current = tf.contrib.layers.batch_norm(
                    current,
                    scale=False,
                    is_training=is_training,
                    scope=name,
                    reuse=reuse
                )
            name = 'Dage_fc2'
            current = fc(
                input_vector=current,
                num_output_length=202,
                name=name,
                reuse=reuse
            )
            current = lrelu(current)
            if enable_bn:
                name = 'Dage_bn2'
                current = tf.contrib.layers.batch_norm(
                    current,
                    scale=False,
                    is_training=is_training,
                    scope=name,
                    reuse=reuse
                )
            name = 'Dage_fc3'
            current = fc(
                input_vector=current,
                num_output_length=101,
                name=name,
                reuse=reuse
            )
            current = lrelu(current)
            if enable_bn:
                name = 'Dage_bn3'
                current = tf.contrib.layers.batch_norm(
                    current,
                    scale=False,
                    is_training=is_training,
                    scope=name,
                    reuse=reuse
                )
            name = 'Dage_fc4'
            current = fc(
                input_vector=current,
                num_output_length=50,
                name=name,
                reuse=reuse
            )
            current = lrelu(current)
            if enable_bn:
                name = 'Dage_bn4'
                current = tf.contrib.layers.batch_norm(
                    current,
                    scale=False,
                    is_training=is_training,
                    scope=name,
                    reuse=reuse
                )
            name = 'Dage_fc5'
            current = fc(
                input_vector=current,
                num_output_length=25,
                name=name,
                reuse=reuse
            )
            current = lrelu(current)
            if enable_bn:
                name = 'Dage_bn5'
                current = tf.contrib.layers.batch_norm(
                    current,
                    scale=False,
                    is_training=is_training,
                    scope=name,
                    reuse=reuse
                )
            name = 'Dage_fc6'
            current = fc(
                input_vector=current,
                num_output_length=12,
                name=name,
                reuse=reuse
            )
            current = lrelu(current)
            if enable_bn:
                name = 'Dage_bn6'
                current = tf.contrib.layers.batch_norm(
                    current,
                    scale=False,
                    is_training=is_training,
                    scope=name,
                    reuse=reuse
                )
            name = 'Dage_fc7'
            current = fc(
                input_vector=current,
                num_output_length=1,
                name=name,
                reuse=reuse
            )
            # output
            return tf.nn.sigmoid(current), current

    def save_checkpoint(self):
        checkpoint_dir = self.chk_dir  # os.path.join(self.chk_dir, 'checkpoint')
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        self.saver.save(
            sess=self.session,
            save_path=os.path.join(checkpoint_dir, 'model'),
            global_step=self.EG_global_step.eval()
        )

    def load_checkpoint(self, model_path=None, chknum=None):
        print("\n\tLoading pre-trained model ...")
        checkpoint_dir = os.path.join(self.save_dir, 'checkpoint', model_path)
        checkpoints_name = 'model-' + str(chknum)
        if checkpoints_name:
            try:
                self.saver.restore(self.session, os.path.join(checkpoint_dir, checkpoints_name))
                return True
            except:
                return False
        else:
            return False

    def sample(self, id_ref, age_ref, name):
        sample_dir = self.smp_dir  # os.path.join(self.smp_dir, 'samples')
        if not os.path.exists(sample_dir):
            os.makedirs(sample_dir)
        _, _, G = self.session.run(
            [self.z_id_I, self.z_age_A, self.G],
            feed_dict={
                self.input_image_id: id_ref,
                self.input_image_age: age_ref
            }
        )
        size_frame = int(np.sqrt(self.size_batch))
        save_batch_images(
            batch_images=G,
            save_path=os.path.join(sample_dir, name),
            image_value_range=self.image_value_range,
            size_frame=[size_frame, size_frame]
        )

    def test(self, id_ref, age_ref, name, size_frame):
        test_dir = self.test_dir  # os.path.join(self.test_dir, 'test')
        if not os.path.exists(test_dir):
            os.makedirs(test_dir)

        _, G = self.session.run(
            [self.z_id_I, self.G],
            feed_dict={
                self.input_image_id: id_ref,
                self.input_image_age: age_ref
            }
        )

        save_batch_images(
            batch_images=G,
            save_path=os.path.join(test_dir, name),
            image_value_range=self.image_value_range,
            size_frame=[size_frame, size_frame]
        )

    def custom_test(self, testing_samples_dir, load_dir, chknum):
        if not self.load_checkpoint(load_dir, chknum):
            print("\tFAILED >_<!")
            exit(0)
        else:
            print("\tSUCCESS ^_^")

        num_samples = int(np.sqrt(self.size_batch))
        file_names = glob(testing_samples_dir)
        if len(file_names) < num_samples:
            print 'The number of testing images is must larger than %d' % num_samples
            exit(0)
        sample_files = file_names[0:num_samples]
        sample = [load_image(
            image_path=sample_file,
            image_size=self.size_image,
            image_value_range=self.image_value_range,
            is_gray=(self.num_input_channels == 1),
        ) for sample_file in sample_files]
        if self.num_input_channels == 1:
            images = np.array(sample).astype(np.float32)[:, :, :, None]
        else:
            images = np.array(sample).astype(np.float32)
        gender_male = np.ones(
            shape=(num_samples, 2),
            dtype=np.float32
        ) * self.image_value_range[0]
        gender_female = np.ones(
            shape=(num_samples, 2),
            dtype=np.float32
        ) * self.image_value_range[0]
        for i in range(gender_male.shape[0]):
            gender_male[i, 0] = self.image_value_range[-1]
            gender_female[i, 1] = self.image_value_range[-1]

        self.test(images, gender_male, 'test_as_male.png', size_frame=num_samples)
        self.test(images, gender_female, 'test_as_female.png', size_frame=num_samples)

        print '\n\tDone! Results are saved as %s\n' % os.path.join(self.save_dir, 'test', 'test_as_xxx.png')

    def custom_test_new(self, testing_samples_dir, load_dir, chknum):
        # chknum = 40200;
        if not self.load_checkpoint(load_dir, chknum):
            print("\tFAILED >_<!")
            exit(0)
        else:
            print("\tSUCCESS ^_^")

        file_names = glob(os.path.join('../DataSet', testing_samples_dir))

        num = 10
        tot = int(len(file_names) / num)

        gender_male = np.ones(
            shape=(num, 2),
            dtype=np.float32
        ) * self.image_value_range[0]
        gender_female = np.ones(
            shape=(num, 2),
            dtype=np.float32
        ) * self.image_value_range[0]
        for i in range(gender_male.shape[0]):
            gender_male[i, 0] = self.image_value_range[-1]
            gender_female[i, 1] = self.image_value_range[-1]

        labels = np.arange(num)
        labels = np.repeat(labels, num)
        query_labels = np.ones(
            shape=(num ** 2, num),
            dtype=np.float32
        ) * self.image_value_range[0]
        for i in range(query_labels.shape[0]):
            query_labels[i, labels[i]] = self.image_value_range[-1]

        for i in range(0, tot):
            sample_files = file_names[i * num:(i + 1) * num]
            test_dir = 'save/' + 'last-test-' + str(chknum)
            test_dir0 = test_dir + '/male/' + str(i)
            test_dir1 = test_dir + '/female/' + str(i)
            if not os.path.exists(test_dir0):
                os.makedirs(test_dir0)
            if not os.path.exists(test_dir1):
                os.makedirs(test_dir1)
            save_sample_file(sample_files, test_dir)

            sample = [load_image(
                image_path=sample_file,
                image_size=self.size_image,
                image_value_range=self.image_value_range,
                is_gray=(self.num_input_channels == 1),
            ) for sample_file in sample_files]

            if self.num_input_channels == 1:
                images = np.array(sample).astype(np.float32)[:, :, :, None]
            else:
                images = np.array(sample).astype(np.float32)

            testname0 = 'test-male' + str(i) + '.png'
            testname1 = 'test-female' + str(i) + '.png'
            inputname = 'input' + str(i) + '.png'

            size_sample = images.shape[0]

            query_images = np.tile(images, [num, 1, 1, 1])
            query_male_gender = np.tile(gender_male, [num, 1])
            query_female_gender = np.tile(gender_female, [num, 1])

            z, G = self.session.run(
                [self.z_id_I, self.G],
                feed_dict={
                    self.input_image_id: query_images,
                    self.input_image_age: query_labels,
                    self.gender: query_male_gender
                }
            )

            save_seperate_images(
                batch_images=G,
                save_path=os.path.join(test_dir0, testname0),
                image_value_range=self.image_value_range,
                size_frame=[size_sample, size_sample]
            )

            z, G = self.session.run(
                [self.z_id_I, self.G],
                feed_dict={
                    self.input_image_id: query_images,
                    self.input_image_age: query_labels,
                    self.gender: query_female_gender
                }
            )

            save_seperate_images(
                batch_images=G,
                save_path=os.path.join(test_dir1, testname1),
                image_value_range=self.image_value_range,
                size_frame=[size_sample, size_sample]
            )

            save_batch_images(
                batch_images=query_images,
                save_path=os.path.join(test_dir, inputname),
                image_value_range=self.image_value_range,
                size_frame=[size_sample, size_sample]
            )

        print '\n\tDone! Results are saved as %s\n' % os.path.join(self.save_dir, 'test', 'test_as_xxx.png')

    def proc_age(self, input):
        ind = tf.argmax(input, 1)
        tmp = tf.one_hot(indices=ind, depth=101, on_value=1, off_value=0)
        output = tf.stack([tf.reduce_sum(tmp[:, 0:6], axis=1),
                           tf.reduce_sum(tmp[:, 6:11], axis=1),
                           tf.reduce_sum(tmp[:, 11:16], axis=1),
                           tf.reduce_sum(tmp[:, 16:21], axis=1),
                           tf.reduce_sum(tmp[:, 21:31], axis=1),
                           tf.reduce_sum(tmp[:, 31:41], axis=1),
                           tf.reduce_sum(tmp[:, 41:51], axis=1),
                           tf.reduce_sum(tmp[:, 51:61], axis=1),
                           tf.reduce_sum(tmp[:, 61:71], axis=1),
                           tf.reduce_sum(tmp[:, 71:], axis=1)], axis=1)
        a0 = tf.constant(-1, shape=[self.size_batch, self.num_categories])
        output = tf.where(tf.equal(output, 0), a0, 2 + a0) #convert (0,1) one-hot-vector to (-1,1)
        return output
