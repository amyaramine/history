# -*- coding: utf-8 -*-
import Models
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from keras.utils import np_utils
import glob
import cPickle as pickle
from keras.callbacks import ModelCheckpoint

img_rows = 100
img_cols = 100
color_type = 1
batch_size = 256
nb_epoch = 50
split = 0.2
nb_classes = 2

np.random.seed(2017)  # for reproducibility

path = '../TepOesophageNewVersion/Input/'
liste = os.listdir(path)

prediction = open('../TepOesophageNewVersion/Prediction.txt').read()
prediction = prediction.split()

i = 0
while(i < 10):
    data_train_all, data_test, target_train_all , target_test = train_test_split(liste, prediction, test_size=split, random_state=i)
    data_train, data_validate, target_train, target_validate = train_test_split(data_train_all, target_train_all, test_size=split, random_state=i)
    X_train = []
    Y_train = []
    X_test = []
    Y_test = []
    X_validate = []
    Y_validate = []
    Y_result = []
    j = 0
    #indices = '-25','0','25'
    # print target_train
    print "Load training data"
    for element in data_train:
        # listeTemp = sorted(os.listdir(path+element+'/Original/'), key=lambda x:int(x.split('.')[0]))
        # print listeTemp

        for elet in sorted(os.listdir(path+element+'/Original/'), key=lambda x:int(x.split('.')[0])):
            tumeur = np.load(path+element+'/Original/' + elet)
            X_train.append(tumeur)
            Y_train.append(target_train[j])

       
#        for folder in sorted(os.listdir(path + element + '/Rotation/'), key=lambda x:int(x.split('.')[0])):
 #           for elet in os.listdir(path + element + '/Rotation/' + folder):
  #              tumeur = np.load(path + element + '/Rotation/' + folder+ '/' + elet)
   #             X_train.append(tumeur)
    #            Y_train.append(target_train[j])

  #      for folder in sorted(os.listdir(path+element+'/Shifted/'), key=lambda x:int(x.split('.')[0])):
  #          for elet in os.listdir(path + element + '/Shifted/' + folder):
   #             tumeur = np.load(path + element + '/Shifted/' + folder+ '/' + elet)
    #            X_train.append(tumeur)
    #            Y_train.append(target_train[j])
        j += 1
    j = 0


    print "Load validation data"
    for element in data_validate:
        # listeTemp = sorted(os.listdir(path+element+'/Original/'), key=lambda x:int(x.split('.')[0]))
        # print listeTemp

        for elet in sorted(os.listdir(path+element+'/Original/'), key=lambda x:int(x.split('.')[0])):
            tumeur = np.load(path+element+'/Original/' + elet)
            X_validate.append(tumeur)
            Y_validate.append(target_validate[j])

  #      for folder in sorted(os.listdir(path + element + '/Rotation/'), key=lambda x:int(x.split('.')[0])):
  #          for elet in os.listdir(path + element + '/Rotation/' + folder):
  #              tumeur = np.load(path + element + '/Rotation/' + folder+ '/' + elet)
  #              X_validate.append(tumeur)
  #              Y_validate.append(target_validate[j])

 #       for folder in sorted(os.listdir(path+element+'/Shifted/'), key=lambda x:int(x.split('.')[0])):
  #          for elet in os.listdir(path + element + '/Shifted/' + folder):
  #              tumeur = np.load(path + element + '/Shifted/' + folder+ '/' + elet)
  #              X_validate.append(tumeur)
  #              Y_validate.append(target_validate[j])
        j += 1
    j = 0

    print "Load test data"
    for element in data_test:
        for elet in sorted(os.listdir(path + element + '/Original'), key=lambda x:int(x.split('.')[0])):
            tumeur = np.load(path + element + '/Original/' + elet)
            X_test.append(tumeur)
            Y_test.append(target_test[j])
        # Y_result.append(target_test)
        j += 1

    X_train = np.asarray(X_train)
    X_train /= 30
    X_train = shuffle(X_train, random_state=0)
    X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
    print "Train : ",X_train.shape

    X_validate = np.asarray(X_validate)
    X_validate /= 30
    X_validate = shuffle(X_validate, random_state=0)
    X_validate = X_validate.reshape(X_validate.shape[0], 1, img_rows, img_cols)
    print "Validation : ", X_validate.shape

    print "Nombre de patients repondeurs apprentissage : ", len(np.where(np.asarray(target_train, dtype=np.int32)>0)[0])
    print "Nombre de patients pour non repondeurs apprentissage : ", len(np.where(np.asarray(target_train, dtype=np.int32) == 0)[0])
    print "Nombre de coupes pour repondeurs apprentissage : ", len(np.where(np.asarray(Y_train, dtype=np.int32)>0)[0])
    print "Nombre de coupes pour non repondeurs apprentissage : ", len(np.where(np.asarray(Y_train, dtype=np.int32) == 0)[0])

    print "\n"
    print "Nombre de patients repondeurs test : ", len(np.where(np.asarray(target_test, dtype=np.int32)>0)[0])
    print "Nombre de patients non repondeurs test : ", len(np.where(np.asarray(target_test, dtype=np.int32) == 0)[0])
    print "Nombre de coupes pour repondeurs test : ", len(np.where(np.asarray(Y_test, dtype=np.int32)>0)[0])
    print "Nombre de coupes pour non repondeurs test : ", len(np.where(np.asarray(Y_test, dtype=np.int32) == 0)[0])
    print "\n"
    # os.system('pause')

    Y_train = np.asarray(Y_train, dtype=np.int32)
    Y_train = np_utils.to_categorical(Y_train, nb_classes)
    Y_train = shuffle(Y_train, random_state=0)


    Y_validate = np.asarray(Y_validate, dtype=np.int32)
    Y_validate = np_utils.to_categorical(Y_validate, nb_classes)
    Y_validate = shuffle(Y_validate, random_state=0)

    X_test = np.asarray(X_test)
    X_test /= 30
    X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)

    Y_test = np.asarray(Y_test, dtype=np.int32)
    Y_test = np_utils.to_categorical(Y_test, nb_classes)

    model = Models.CNN_2Couches(img_rows, img_cols,1)
    checkpointer = ModelCheckpoint(filepath="weights.hdf5", verbose=1, save_best_only=True)
    model.fit(X_train, Y_train, batch_size=batch_size, epochs=nb_epoch,
              verbose=1, validation_data=(X_validate, Y_validate), shuffle=True, callbacks=[checkpointer])
    #with open('history', 'wb') as fp:
    #    pickle.dump(history.history, fp)
    model.load_weights('weights.hdf5')
    score = model.evaluate(X_test, Y_test, verbose=0)
    print "score = ", score
    results = model.predict_classes(X_test)

    k = 0

    listePrediction = []
    for elet in data_test:
        liste2 = glob.glob("../TepOesophageNewVersion/Input/" + elet + '/Original/*')
        j = 0
        PredictionReponse = 0
        while (j < len(liste2)):
            PredictionReponse += results[k]
            k += 1
            j += 1
        PredictionReponse = float(PredictionReponse) / float(j)
     #   print PredictionReponse
        if (PredictionReponse > 0.5):
            listePrediction.append(1)
        else:
            listePrediction.append(0)


    accuracy = 0
    indexPrediction = 0
    while (indexPrediction < len(listePrediction)):
        if (listePrediction[indexPrediction] == int(target_test[indexPrediction])):
            accuracy += 1
        indexPrediction += 1

    accuracy = float(accuracy) / float(len(listePrediction))
    print "Accuracy :", accuracy


    VP = 0
    FN = 0
    FP = 0
    VN = 0
    indexPrediction = 0
    while (indexPrediction < len(listePrediction)):
        # print "listePrediction = ", listePrediction[indexPrediction]
        # print "Y_result = ", Y_result[indexPrediction]
        if ((listePrediction[indexPrediction] == 1) & (int(target_test[indexPrediction]) == 1)):
            VP += 1
        elif ((listePrediction[indexPrediction] == 0) & (int(target_test[indexPrediction]) == 1)):
            FN += 1
        elif ((listePrediction[indexPrediction] == 0) & (int(target_test[indexPrediction]) == 0)):
            VN += 1
        else:
            FP += 1
        indexPrediction += 1

    sens = float(VP) / float(float(VP) + float(FN))
    spec = float(VN) / float(float(VN) + float(FP))

    print "Confusion Matrix"
    print "            Malade      Non Malade "
    print "TestP     ", VP, "        ", FP
    print "TestN     ", FN, "        ", VN
    print "**************************************", VP + FP + FN + VN
    print "Sensibilite = ", sens
    print "Specificite = ", spec

    i += 1

