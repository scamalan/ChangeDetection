""" E-ReCNN Change Detection Model

This script allows the user to train the E-ReCNN model to detect 
changes on GEE images with Leave-One_Region-Out cross-validation. 
For each region, the model is trained and the region is tested.

The model inputs bi-temporal images from the same region, and outputs 
the change map of the mining ponds of the region. From the image folder,
the system loads the images according to the bands that requested and
feed the model with image and label patches.

This script requires that `numpy, scipy, sklearn, cv2, tensorflow, matplotlib` 
be installed within the Python environment you are running this script in.

It also requires that 'read_data and load_data' modules which are stated 
in this folder.

This file can also be imported as a module and contains the following
functions:

    * main - the main function of the script

    Parameters
    ----------
    directory : str
        The images of the region directory
    run : str
        A binary of multiclass classification selection
    channels : int
        Number of channels of the images that the model is trained with

"""

import os
import sys
import datetime
import logging
import argparse

from read_data_aug_fld import *
from load_data import *

import numpy as np
import scipy.io

from sklearn.utils import class_weight
from sklearn.preprocessing import OneHotEncoder

import tensorflow as tf

import cv2

import matplotlib.pyplot as plt

if __name__ == "__main__":
    
    parse = argparse.ArgumentParser()
    parse.add_argument(
        "-d",
        "--directory",
        dest="directory",
        help="Directory to run splitting on",
        default="C:\\Users\\camalas.DEACNET\\Project\\Mining\\images\\Test_diff\\Diff_ponds\\"#"C:\\Users\\camalas.DEACNET\\Project\\Mining\\images\\Bigger_Set_21_19\\"#os.path.join("data", "processed"),
    )
    parse.add_argument(
        "-td",
        "--test_directory",
        dest="test_directory",
        help="Directory to run splitting on",
        default="C:\\Users\\camalas.DEACNET\\Project\\Mining\\images\\Bigger_Set_21_19\\"#"C:\\Users\\camalas.DEACNET\\Project\\Mining\\images\\Test_diff\\Diff_ponds\\"#None,
    )
    parse.add_argument(
        "-r", "--run", dest="run", help="Use binary or multiclass", default="binary"#"binary"
    )
    parse.add_argument(
        "-c",
        "--channels",
        dest="channels",
        help="Number of channels to use",
        default=3,
        type=int,
    )
    parse.add_argument(
        "-t1",
        "--time1",
        dest="time1",
        help="First time to look the difference",
        default="2019-08-18", #None,
        type=str,
    )
    parse.add_argument(
        "-t2",
        "--time2",
        dest="time2",
        help="Second time to look the difference",
        default="2021-07-23", #None,
        type=str,
    )
    parse.add_argument(
        "-tt1",
        "--time1_test",
        dest="time1_test",
        help="First time to look the difference",
        default="2019-08-18", #None,
        type=str,
    )
    parse.add_argument(
        "-tt2",
        "--time2_test",
        dest="time2_test",
        help="Second time to look the difference",
        default="2021-07-23", #None,
        type=str,
    )
    parse.add_argument(
        "-o", "--output", dest="output", help="Output directory", default="", type=str
    )
    parse.add_argument(
        "-l",
        "--log-file",
        dest="log_file",
        help="Log file name",
        default=None,
        type=str,
    )
    parse.add_argument(
        "-t",
        "--tf-log-dir",
        dest="tf_log_dir",
        help="TensorFlow log file name",
        default=None,
        type=str,
    )
    args = parse.parse_args()
    
    reg = ["1","2","3","4","5","6","7","8","9","10","11","12","13","14","15", "16"]
    # Set output directories
    FOLDER_NAME = str(args.directory)
    
    if args.test_directory is None:
        TEST_FOLDER_NAME = str(args.directory)
        time1_test = args.time1 
        time2_test = args.time2
    else:
        TEST_FOLDER_NAME = str(args.test_directory)
        time1_test = args.time1_test 
        time2_test = args.time2_test
        
    
    enc = OneHotEncoder()

    # Create and configure logger
    if args.log_file is None:
        logging.basicConfig(
            filename=os.path.join(str(args.output), "run.log"),
            format="%(asctime)s %(message)s",
            filemode="w",
        )
    else:
        logging.basicConfig(
            filename=str(args.output), format="%(asctime)s %(message)s", filemode="w"
        )

    # Creating an object
    logger = logging.getLogger()
    flag = True
    
    for test_im in reg:
         
        logger.info("Test Image: %s", str(test_im))
        if args.test_directory is None:
            test_im_arg = test_im
            flag =True
        elif test_im == "1":
            test_im_arg = "0"
            flag = True
        else:
            test_im_arg = "0"
            flag = False
            
        if (flag):
            
            logger.info("Begin data loading")
            (X_train1, X_train2, y_train) = create_train_set_from_folder(   #read_data_aug_fld.
                FOLDER_NAME, test_im_arg, args.channels, args.run, args.time1, args.time2 
            )
            logger.info("Data loading finished")
    
            class_weights = class_weight.compute_class_weight(
                class_weight="balanced", classes=np.unique(y_train), y=y_train
            )
            weight_dict = {}
    
            for i in np.unique(y_train):
                weight_dict[i] = class_weights[i]
    
            if args.run == "multiclass":
                y_train = tf.keras.utils.to_categorical(y_train)
    
            # Begin model definition
    
            net_input1 = tf.keras.layers.Input(
                shape=(X_train1.shape[1:]), name="net_input1"
            )
            net_input2 = tf.keras.layers.Input(shape=(X_train2.shape[1:]))
    
            FEAT_NUM = 32
            initializer = tf.keras.initializers.glorot_uniform()
    
            '''net_input1 = tf.keras.layers.Input(
                shape=(X_train1.shape[1:]), name="net_input1"
            )
            net_input2 = tf.keras.layers.Input(shape=(X_train2.shape[1:]))
    
            FEAT_NUM = 32
            initializer = tf.keras.initializers.glorot_uniform()'''
    
            # the first branch operates on the first input
            first_nn = tf.keras.layers.Conv2D(
                FEAT_NUM,
                (3, 3),
                activation="relu",
                dilation_rate=2,
                kernel_initializer=initializer,
            )(
                net_input1
            )  # X_train1
            first_mdl = tf.keras.Model(inputs=net_input1, outputs=first_nn)
    
            # the second branch opreates on the second input
            second_nn = tf.keras.layers.Conv2D(
                FEAT_NUM,
                (3, 3),
                activation="relu",
                dilation_rate=2,
                kernel_initializer=initializer,
            )(
                net_input2
            )  # X_train2)
            second_mdl = tf.keras.Model(inputs=net_input2, outputs=second_nn)
    
            timestep_feature = tf.keras.layers.Concatenate(axis=1)([first_nn, second_nn])
    
            timestep_feature = tf.reshape(timestep_feature, [-1, 2, FEAT_NUM])
    
            lstm = tf.keras.layers.LSTM(
                128,
                input_shape=(timestep_feature.shape),
                return_sequences=True,
                kernel_initializer=initializer,
            )(timestep_feature)
            lstm = tf.keras.layers.Dropout(0.2)(lstm)
    
            lstm = tf.keras.layers.LSTM(128, kernel_initializer=initializer)(lstm)
    
            net_output = tf.keras.layers.Dense(
                FEAT_NUM, activation="relu", kernel_initializer=initializer
            )(lstm)
            if args.run == "binary":
                net_output = tf.keras.layers.Dense(1,activation='sigmoid')(net_output)
            elif args.run == "multiclass":
                net_output = tf.keras.layers.Dense(4, activation="softmax")(net_output)
            else:
                logger.error("Error: Invalid run type selected - %s", str(args.run))
            # End model definition
    
            model = tf.keras.Model(inputs=[net_input1, net_input2], outputs=[net_output])
            model.summary()
            metrics = [
                "accuracy",
                tf.keras.metrics.Precision(),
                tf.keras.metrics.Recall(),
                tf.keras.metrics.TruePositives(),
                tf.keras.metrics.TrueNegatives(),
                tf.keras.metrics.FalsePositives(),
                tf.keras.metrics.FalseNegatives(),
            ]
            
            if args.run == "binary":
                model.compile(
                    loss="binary_crossentropy",
                    optimizer=tf.keras.optimizers.Nadam(
                        learning_rate=0.001,
                        beta_1=0.9,
                        beta_2=0.999,
                        epsilon=1e-07,
                        name="Nadam",
                    ),
                    metrics=metrics,
                )
            elif args.run == "multiclass":
                model.compile(
                    loss="categorical_crossentropy",
                    optimizer=tf.keras.optimizers.Nadam(
                        learning_rate=0.001,
                        beta_1=0.9,
                        beta_2=0.999,
                        epsilon=1e-07,
                        name="Nadam",
                    ),
                    metrics=metrics,
                )
            else:
                # error
                logger.error("Error: Invalid run type selected - %s", str(args.run))
                sys.exit(1)
            
    
            checkpoint_path = os.path.join(
                str(args.output),
                "training_" + str(test_im),
                "Combined_lbl_Pxl_wise_75epc.ckpt",
            )
    
            # Create a callback that saves the model's weights
            cp_callback = tf.keras.callbacks.ModelCheckpoint(
                filepath=checkpoint_path, save_weights_only=True, verbose=1
            )
            if args.tf_log_dir is None:
                LOG_DIR = os.path.join(
                    "logs",
                    "fit",
                    str(test_im),
                    datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),
                )
            else:
                LOG_DIR = str(args.tf_log_dir)
    
            tensorboard_callback = tf.keras.callbacks.TensorBoard(
                log_dir=LOG_DIR, histogram_freq=1
            )
    
            history = model.fit(
                [X_train1, X_train2],
                y_train,
                epochs=1,
                validation_split=0.3,
                verbose=2,
                callbacks=[cp_callback, tensorboard_callback],
                batch_size=256,
            )
    
            logger.info(history.history.keys())
            # summarize history for accuracy
            plt.plot(history.history["accuracy"])
            plt.plot(history.history["val_accuracy"])
            plt.title(test_im + "model accuracy")
            plt.ylabel("accuracy")
            plt.xlabel("epoch")
            plt.legend(["train", "test"], loc="upper left")
            plt.show()
            plt.savefig(test_im + "_Model_Accuracy_fig.png")
            # summarize history for loss
            plt.plot(history.history["loss"])
            plt.plot(history.history["val_loss"])
            plt.title("model loss")
            plt.ylabel("loss")
            plt.xlabel("epoch")
            plt.legend(["train", "test"], loc="upper left")
            plt.show()
            plt.savefig(test_im + "_Model_Loss_fig.png")
            
        
                    
        # Load images
        if args.channels == 3:
            image1_t =cv2.imread(
                os.path.join(
                    TEST_FOLDER_NAME,
                    test_im,
                    "R" + str(test_im) + "_original_"+time1_test+".tif",
                )
            )
            image1_t = np.asarray(image1_t)
            image2_t = cv2.imread(
                os.path.join(
                    TEST_FOLDER_NAME,
                    test_im,
                    "R" + str(test_im) + "_original_"+time2_test+".tif",
                )
            )
            image2_t = np.asarray(image2_t)
        elif args.channels in [6, 10]:
            image1_t = scipy.io.loadmat(
                os.path.join(
                    TEST_FOLDER_NAME,
                    test_im,
                    "R"
                    + str(test_im)
                    + "_original_"+time1_test+"_ch"
                    + str(args.channels)
                    + ".mat",
                )
            )
            image2_t = scipy.io.loadmat(
                os.path.join(
                    TEST_FOLDER_NAME,
                    test_im,
                    "R"
                    + str(test_im)
                    + "_original_"+time2_test+"_ch"
                    + str(args.channels)
                    + ".mat",
                )
            )
            image1_t = image1_t["new"]
            image2_t = image2_t["new"]
        else:
            # error
            logger.error(
                "Error: Invalid number of channels selected - %s", str(args.channels)
            )
            sys.exit(1)

        # Load labels
        if args.run == "binary":
            label_t = cv2.imread(
                os.path.join(
                    TEST_FOLDER_NAME,
                    test_im,
                    "R" + str(test_im) + "_original_Binary_change_thr.png",
                )
            )
        elif args.run == "multiclass":
            label_t = cv2.imread(
                os.path.join(
                    TEST_FOLDER_NAME,
                    test_im,
                    "R" + str(test_im) + "_original_four_change_thr.png",
                )
            )
        else:
            # error
            logger.error("Error: Invalid run type selected - %s", str(args.run))
            sys.exit(1)

        label_t = np.asarray(label_t)

        X_test1, X_test2, y_test, idx_list = create_train_set_seperate(    #read_data_aug_fld.
                                                                        image1_t,
                                                                        image2_t,
                                                                        label_t,
                                                                        args.channels
                                                                        )

        X_test = [X_test1, X_test2]
        if args.run == "multiclass":
            y_test = tf.keras.utils.to_categorical(y_test)

        # Evaluate the model on the test data using `evaluate`
        logger.info("Evaluate on test data")
        results = model.evaluate(X_test, y_test, batch_size=256)
        logger.info("test loss, test acc: %s", str(results))

        # Generate predictions (probabilities -- the output of the last layer)
        # on new data using `predict`
        logger.info("Generate predictions for a sample")
        predictions_1 = model.predict(X_test)
        predictions = np.squeeze(np.array(predictions_1))
        logger.info("predictions shape: %s", str(predictions.shape))

        prd_lbl = np.round(predictions)
        row, col, s = label_t.shape
        old_lbl = create_change_label_pixel(idx_list, y_test, row, col, args.run)
        new_lbl = create_change_label_pixel(idx_list, prd_lbl, row, col, args.run)
        cv2.imwrite(
            os.path.join(str(args.output), "Predicted_change_" + test_im + ".png"),
            new_lbl,
        )
        cv2.imwrite(
            os.path.join(str(args.output), "Predicted_grdtrh_" + str(test_im) + ".png"),
            old_lbl,
        )
