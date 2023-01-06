import warnings
warnings.filterwarnings('ignore')

from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import SGD

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from art.estimators.classification import TensorFlowV2Classifier
from art.attacks.evasion import BasicIterativeMethod
from art.defences.trainer import AdversarialTrainer
from art.attacks.evasion import FastGradientMethod

def build(shape, classes,only_digits=True):
    model_1= Sequential()
    model_1.add(Flatten())
    model_1.add(Dense(1024,activation=('relu'),input_dim=512))
    model_1.add(Dense(512,activation=('relu')))
    model_1.add(Dense(256,activation=('relu')))
    model_1.add(Dense(128,activation=('relu')))
    model_1.add(Dense(10,activation=('softmax')))
    return model_1

def load_dataset_cifar():
    # load dataset
    (trainX, trainY), (testX, testY) = cifar10.load_data()
    # one hot encode target values
    trainY = to_categorical(trainY)
    testY = to_categorical(testY)
    return np.array(trainX/255.0, dtype = np.float32), trainY, np.array(testX/255.0, dtype = np.float32), np.array(testY)

class MyTfClassifier(TensorFlowV2Classifier):
  def save_model(self, name):
    self.model.save(name)

def train_step(model, images, labels):
    loss_object = tf.keras.losses.CategoricalCrossentropy()
    lr = 0.01
    comms_round = 1
    optimizer = SGD(lr=lr,
                    decay=lr / comms_round,
                    momentum=0.9
                    )

    with tf.GradientTape() as tape:
        predictions = model(images, training=True)
        loss = loss_object(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))

def model_pre_adverserial(model_weight):
    comms_round = 1
    lr = 0.01
    loss_object = tf.keras.losses.CategoricalCrossentropy()
    metrics = ['accuracy']
    optimizer = SGD(lr=lr,
                    decay=lr / comms_round,
                    momentum=0.9
                    )

    model = tf.keras.models.load_model(model_weight)
    classifier = MyTfClassifier(
        model=model,
        loss_object=loss_object,
        train_step=train_step,
        nb_classes=10,
        input_shape=(28, 28, 1),
        clip_values=(0, 1)
    )
    return classifier

class MyAdvTrainer(AdversarialTrainer):
  def save_model(self, name):
    print(self._classifier)
    # self._classifier.save(name)
    # tf.keras.models.save_model(
    #     self._classifier,
    #     name,
    #     overwrite=True,
    #     include_optimizer=True)

    # self._classifier.save_weights(name)
    self._classifier.save_model(name)

  def load_model(self, name):
    classifier = model_pre_adverserial(name)
    self._classifier = classifier

def adverserial_train(save_weight, load_weight, classifier_weight, iter_no, epoch_no):
    cifar_train, cifar_label_train, cifar_test, cifar_label_test = load_dataset_cifar()
    cifar_train, cifar_valid, cifar_label_train, cifar_label_valid = train_test_split(cifar_train, cifar_label_train,
                                                                                      test_size=0.2, random_state=42)
    classifier = model_pre_adverserial(classifier_weight)

    # Here is the command we had used for the Adversarial Training
    attacks = BasicIterativeMethod(classifier, eps=0.3, eps_step=0.01, max_iter=iter_no)
    trainer = MyAdvTrainer(classifier, attacks, ratio=1.0)

    # load trainer
    if len(load_weight)>0:
        trainer.load_model(load_weight)

    # Trainer fit and save
    trainer.fit(cifar_train, cifar_label_train, nb_epochs=epoch_no, batch_size=50)
    trainer.save_model(save_weight)

def perplexity_medium_fixed(predictions, test):
  perplexity = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
  for i, pred in enumerate(list(predictions)):
    if np.argmax(pred) == np.argmax(test[i]):
      perplexity[np.argmax(pred)]=perplexity[np.argmax(pred)] + np.log2(pred[np.argmax(pred)])*100
  exposure = - perplexity
  return exposure

def adverserial_predict(load_weight, classifier_weight):
    cifar_train, cifar_label_train, cifar_test, cifar_label_test = load_dataset_cifar()

    classifier = model_pre_adverserial(classifier_weight)
    # Here is the command we had used for the Adversarial Training
    attacks = BasicIterativeMethod(classifier, eps=0.3, eps_step=0.01, max_iter=1000)
    trainer = MyAdvTrainer(classifier, attacks, ratio=1.0)

    # load trainer
    trainer.load_model(load_weight)
    attacker = FastGradientMethod(classifier, eps=0.5)
    x_test_adv = attacker.generate(cifar_test[:100])

    # "Adversarial test data (first 100 images): Adverserial Training setup"

    ex = perplexity_medium_fixed(trainer.predict(x_test_adv), cifar_label_test[:100])

    x_test_adv_pred_after_attack = np.argmax(trainer.predict(x_test_adv), axis=1)
    nb_correct_adv_pred = np.sum(x_test_adv_pred_after_attack == np.argmax(cifar_label_test[:100], axis=1))

    print("Adversarial test data (first 100 images): Adverserial Training setup")
    print("Correctly classified: {}".format(nb_correct_adv_pred))
    print("Incorrectly classified: {}".format(100 - nb_correct_adv_pred))

    print(ex/10000)

    # "test data (first 100 images): Adverserial Training setup"

    ex = perplexity_medium_fixed(trainer.predict(cifar_test[:100]), cifar_label_test[:100])

    x_test_adv_pred_after_attack = np.argmax(trainer.predict(cifar_test[:100]), axis=1)
    nb_correct_adv_pred = np.sum(x_test_adv_pred_after_attack == np.argmax(cifar_label_test[:100], axis=1))

    print("test data (first 100 images): Adverserial Training setup")
    print("Correctly classified: {}".format(nb_correct_adv_pred))
    print("Incorrectly classified: {}".format(100 - nb_correct_adv_pred))

    print(ex/10000)

    # "test data (first 100 images): Normal Training setup"

    ex = perplexity_medium_fixed(classifier.predict(cifar_test[:100]), cifar_label_test[:100])

    x_test_adv_pred_after_attack = np.argmax(classifier.predict(cifar_test[:100]), axis=1)
    nb_correct_adv_pred = np.sum(x_test_adv_pred_after_attack == np.argmax(cifar_label_test[:100], axis=1))

    print("test data (first 100 images): Normal Training setup")
    print("Correctly classified: {}".format(nb_correct_adv_pred))
    print("Incorrectly classified: {}".format(100 - nb_correct_adv_pred))

    print(ex / 10000)

    # "Adverserial test data (first 100 images): Normal Training setup"

    ex = perplexity_medium_fixed(classifier.predict(x_test_adv), cifar_label_test[:100])

    x_test_adv_pred_after_attack = np.argmax(classifier.predict(x_test_adv), axis=1)
    nb_correct_adv_pred = np.sum(x_test_adv_pred_after_attack == np.argmax(cifar_label_test[:100], axis=1))

    print("Adverserial test data (first 100 images): Normal Training setup")
    print("Correctly classified: {}".format(nb_correct_adv_pred))
    print("Incorrectly classified: {}".format(100 - nb_correct_adv_pred))

    print(ex / 10000)