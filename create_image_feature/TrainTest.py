import os
import numpy as np
from PIL import Image

from tensorflow.keras.models import Sequential, model_from_json
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    Dropout,
    Flatten,
    Dense,
    Activation,
    RandomBrightness,
    Rescaling,
)
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.optimizers import Adam

# ImageDataGenerator is deprecated
# from keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import image_dataset_from_directory

from create_crops_of_Entire_Image import create_crops_of_entire_Image


def show_all_files_in_directory(input_path):
    "This function reads the path of all files in directory input_path"
    files_list = []
    for path, subdirs, files in os.walk(input_path):
        for file in files:
            if file.endswith(".png"):
                files_list.append(os.path.join(path, file))
    return files_list


def check_and_create(dir_path):
    if os.path.exists(dir_path):
        return True
    else:
        os.makedirs(dir_path)
        return False


def get_models(inputshape=(40, 40, 3), classes=4, lr=0.001):
    """
    Create original classification model
    """

    model = Sequential()
    # first set of CONV => RELU => POOL layers
    model.add(Conv2D(20, (15, 15), padding="same", input_shape=inputshape))
    model.add(Activation("relu"))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation("relu"))
    model.add(Dropout(0.5))

    model.add(Dense(classes))
    model.add(Activation("softmax"))

    return model


def Load_Entire_Image(Entire_Image_path):

    All_jpg = show_all_files_in_directory(Entire_Image_path)
    path_label = []

    for i in All_jpg:
        path = i
        label1 = i.split("/")[-3].split("-")[2]
        label2 = i.split("/")[-3].split("-")[3]
        path_label.append((path, (label1, label2)))

    return path_label


class TrainTest:
    def __init__(
        self, base_path="/home/batool/Directroy/", save_path="/home/batool/Directroy/"
    ):

        self.model = None
        self.base_path = base_path
        self.save_path = save_path

    def add_model(
        self,
        classes,
        model,
        model_path="/home/batool/Directroy/Wall/model/",
    ):

        self.model = model
        self.classes = classes
        model_json = self.model.to_json()
        print("\n*************** Saving New Model Structure ***************")
        with open(os.path.join(model_path, "model.json"), "w") as json_file:
            json_file.write(model_json)
            print("json file written")

    # loading the model structure from json file
    def load_model_structure(
        self,
        classes,
        model_path="/home/batool/per_batch/Wall/model/homegrown_model.json",
    ):

        # reading model from json file
        json_file = open(model_path, "r")
        model = model_from_json(json_file.read())
        json_file.close()

        self.model = model
        self.classes = classes

        return model

    def load_weights(
        self, weight_path="/home/batool/per_batch/Wall/model/weights.02-3.05.hdf5"
    ):

        self.model.load_weights(weight_path)

    def train_model(
        self,
        batch_size,
        data_path="/home/batool/beam_selection/image/data",
        window=50,
        lr=0.002,
        epochs=10,
        model_path="/home/batool/Directroy/Wall/model/",
    ):
        # # ImageDataGenerator is deprecated
        # Train
        # Create an Image Datagenerator model, and normalize
        # traingen = ImageDataGenerator(rescale=1.0 / 255, brightness_range=[0.5, 1.5])
        # train_generator = traingen.flow_from_directory(
        #     data_path + "/train/",
        #     target_size=(window, window),
        #     color_mode="rgb",
        #     batch_size=batch_size,
        #     class_mode="categorical",
        #     shuffle=True,
        # )

        ###########
        train_dataset = image_dataset_from_directory(
            data_path + "/train/",
            image_size=(window, window),
            labels="inferred",
            batch_size=batch_size,
            label_mode="categorical",
            shuffle=True,
        )
        preprocessing = Sequential(
            [
                Rescaling(1.0 / 255),
                RandomBrightness(factor=0.5),  # same as brightness_range=[0.5, 1.5]
            ]
        )
        train_generator = train_dataset.map(lambda x, y: (preprocessing(x), y))
        ###########

        batchX, batchy = train_generator.next()
        print(
            "Batch shape=%s, min=%.3f, max=%.3f"
            % (batchX.shape, batchX.min(), batchX.max())
        )

        STEP_SIZE_TRAIN = train_generator.n // train_generator.batch_size

        # # ImageDataGenerator is deprecated
        # Validation
        # Create an Image Datagenerator model, and normalize
        # valgen = ImageDataGenerator(rescale=1.0 / 255, brightness_range=[0.5, 1.5])
        # validation_generator = valgen.flow_from_directory(
        #     data_path + "/validation/",
        #     target_size=(window, window),
        #     color_mode="rgb",
        #     batch_size=batch_size,
        #     class_mode="categorical",
        #     shuffle=True,
        # )

        ###########
        val_dataset = image_dataset_from_directory(
            data_path + "/validation/",
            image_size=(window, window),
            labels="inferred",
            batch_size=batch_size,
            label_mode="categorical",
            shuffle=True,
        )
        preprocessing = Sequential(
            [
                Rescaling(1.0 / 255),
                RandomBrightness(factor=0.5),  # same as brightness_range=[0.5, 1.5]
            ]
        )
        validation_generator = val_dataset.map(lambda x, y: (preprocessing(x), y))
        ###########

        STEP_SIZE_Validation = validation_generator.n // validation_generator.batch_size

        self.model.compile(
            loss=categorical_crossentropy,
            optimizer=Adam(lr=lr),
            metrics=["accuracy"],
        )
        print("*******************Saving model weights****************")
        self.model.fit_generator(
            train_generator,
            steps_per_epoch=STEP_SIZE_TRAIN,
            validation_data=validation_generator,
            validation_steps=STEP_SIZE_Validation,
            epochs=epochs,
        )

        self.model.save_weights(model_path + "model_weights.hdf5")

    def test_model(
        self,
        batch_size,
        data_path="/home/batool/beam_selection/image/data",
        window=50,
        lr=0.002,
        epochs=10,
        model_path="/home/batool/Directroy/Wall/model/",
    ):

        # # ImageDataGenerator is deprecated
        # testgen = ImageDataGenerator(rescale=1.0 / 255, brightness_range=[0.5, 1.5])
        # test_generator = testgen.flow_from_directory(
        #     data_path + "/test/",
        #     target_size=(window, window),
        #     color_mode="rgb",
        #     batch_size=batch_size,
        #     class_mode="categorical",
        #     shuffle=True,
        # )

        ###########
        test_dataset = image_dataset_from_directory(
            data_path + "/test/",
            image_size=(window, window),
            labels="inferred",
            batch_size=batch_size,
            label_mode="categorical",
            shuffle=True,
        )
        # it's weird apllying data augmentation on test data
        preprocessing = Sequential(
            [
                Rescaling(1.0 / 255),
                RandomBrightness(factor=0.5),  # same as brightness_range=[0.5, 1.5]
            ]
        )
        test_generator = test_dataset.map(lambda x, y: (preprocessing(x), y))
        ###########

        STEP_SIZE_TEST = test_generator.n // test_generator.batch_size

        self.model.compile(
            loss=categorical_crossentropy,
            optimizer=Adam(lr=lr),
            metrics=["accuracy"],
        )
        score = self.model.evaluate_generator(
            test_generator, steps=STEP_SIZE_TEST, verbose=1
        )
        print("Test loss:", score[0])
        print("Test accuracy:", score[1])

        label = test_generator.class_indices
        self.labels = dict((v, k) for k, v in label.items())
        print(self.labels)

    def predict_on_crops(self, entire_images_path, window=50, stride=20):

        # For each image predict ton corps
        for count, each_image_path in enumerate(entire_images_path):

            print("**********Create crops and save to swap**************")
            SWAP = create_crops_of_entire_Image(
                each_image_path,
                "/home/batool/beam_selection/image/swap",
                window,
                stride,
            )
            print("**********Create crops is done**************")

            # # ImageDataGenerator is deprecated
            # predgen = ImageDataGenerator(rescale=1.0 / 255)
            # preds_generator = predgen.flow_from_directory(
            #     SWAP,
            #     target_size=(window, window),
            #     color_mode="rgb",
            #     batch_size=1,
            #     shuffle=False,
            # )
            ###########
            pred_dataset = image_dataset_from_directory(
                SWAP,
                image_size=(window, window),
                labels="inferred",
                batch_size=1,
                # label_mode="categorical",
                shuffle=False,
            )
            preprocessing = Sequential(
                [
                    Rescaling(1.0 / 255),
                ]
            )
            preds_generator = pred_dataset.map(lambda x, y: (preprocessing(x), y))
            ###########

            STEP_SIZE_PRED = preds_generator.n // preds_generator.batch_size
            preds_generator.reset()
            pred = self.model.predict_generator(
                preds_generator, steps=STEP_SIZE_PRED, verbose=1
            )
            print("one image predicted, the pred shape is {}".format(pred.shape))

            # flow from directory sweeps the Images alphabitcly, we need to map each prediction to the right one
            print("**********Maping to the right index**************")
            feeding_order = [
                SWAP + "/" + str(i) + ".png" for i in range(preds_generator.n)
            ]
            feeding_order = sorted(feeding_order)
            # print(feeding_order)
            pred_correct = np.zeros((preds_generator.n, 4), dtype=np.float32)
            for number, value in enumerate(feeding_order):
                # print(value)
                right_index = value.split("/")[-1].split(".")[0]
                pred_correct[int(right_index), :] = pred[number, :]
                # print('the shape of corrected prediction is {}'.format(pred_correct.shape))

            print(pred_correct)
            # find top 3 guesses
            votes = np.argmax(pred_correct, axis=1)
            print(votes)
            print(type(votes))
            print(votes.shape)

            vote_shape = np.transpose(votes.reshape(int((960 - 40) / 5) + 1, -1))
            print(vote_shape)
            print(vote_shape.shape)

            np.save(
                "/home/batool/beam_selection/image/npys/"
                + each_image_path.split("/")[-1].split(".")[0]
                + ".npy",
                vote_shape,
            )

            ######### TO image
            image_to_save = np.zeros(
                (vote_shape.shape[0], vote_shape.shape[1], 3), dtype=np.float32
            )

            for r in range(vote_shape.shape[0]):
                for c in range(vote_shape.shape[1]):
                    if vote_shape[r, c] == 0:
                        # background
                        image_to_save[r, c, :] = (255, 255, 255)
                    elif vote_shape[r, c] == 1:
                        # bus
                        image_to_save[r, c, :] = (255, 0, 0)
                    elif vote_shape[r, c] == 2:
                        # car
                        image_to_save[r, c, :] = (255, 128, 0)
                    elif vote_shape[r, c] == 3:
                        # truck
                        image_to_save[r, c, :] = (51, 153, 255)

            print(image_to_save)

            image_to_save = image_to_save.astype("uint8")

            name = each_image_path.split("/")[-1]
            img = Image.fromarray(image_to_save, mode="RGB")
            img.save("/home/batool/beam_selection/image/prediction/" + name)

            # for c, pixel in enumerate(votes):
            #     if pixel == 0:
            #         #background
            #         image_to_save[c,:] = (255,255,255)
            #     elif pixel == 1:
            #         #bus
            #         image_to_save[c,:] = (255,0,0)
            #     elif pixel ==2:
            #         #car
            #         image_to_save[c,:] = (255,128,0)
            #     elif pixel ==3:
            #         #truck
            #         image_to_save[c,:] = (51,153,255)

            # print(image_to_save)

            # vote_shape = np.transpose(image_to_save.reshape(3,int((960-15)/5)+1,106))
            # print(vote_shape.shape)
            # print(vote_shape[0])

            # img = Image.fromarray(show,mode='RGB')
            # img.save('test2.png')

            # maximum_per_pixel = pred_correct[:,0]
            # decision = maximum_per_pixel.argsort()[-10:][::-1]
            # print('Selected pixels are:',decision)
            # vote = np.zeros(pred_correct.shape[0],)
            # vote [decision] = 1
            # vote_shape = np.transpose(vote.reshape(int((4000-50)/20)+1,-1))
            # path_of_npy_save = '/home/batool/Directroy/predictions/feature_map_npy/'+each_image_path.split('/')[-2]
            # check_and_create(path_of_npy_save)
            # print(each_image_path)
            # print(path_of_npy_save+'/'+each_image_path.split('/')[-1].split('.')[0]+'.npy')
            # np.save(path_of_npy_save+'/'+each_image_path.split('/')[-1].split('.')[0]+'.npy',vote_shape)

            # show= (1-vote_shape).astype('uint8')*255
            # img = Image.fromarray(show,mode='L')

            # path_of_featuremap_save = '/home/batool/Directroy/predictions/feature_map/'+each_image_path.split('/')[-2]
            # check_and_create(path_of_featuremap_save)
            # img.save(path_of_featuremap_save+'/'+each_image_path.split('/')[-1])
            # print('done predction for {}',format(count/float(len(entire_images_path))))
            # print('**********Prediction is done for this example**************')
