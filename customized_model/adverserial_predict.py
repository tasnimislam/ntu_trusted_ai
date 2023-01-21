from customized_model.adverserial_model import *
from art.attacks.evasion import FastGradientMethod
from art.attacks.evasion import BasicIterativeMethod
from customized_model.evaluation_metrics import *
from tensorflow import keras
from tensorflow.keras.optimizers import SGD
import copy

def adverserial_classifier_setup(load_weight, classifier_weight):
    classifier = model_pre_adverserial(classifier_weight)

    # Here is the command we had used for the Adversarial Training
    attacks = BasicIterativeMethod(classifier, eps=0.3, eps_step=0.01, max_iter=1000)
    trainer = MyAdvTrainer(classifier, attacks, ratio=1.0)

    # load trainer
    trainer.load_model(load_weight)
    return trainer, classifier

def predict_in_general(trainer, string, classifier):
    cifar_train, cifar_label_train, cifar_test, cifar_label_test = load_dataset_cifar()

    attacker = FastGradientMethod(classifier, eps=0.5)
    x_test_adv = attacker.generate(cifar_test[:100])

    ex = perplexity_medium_fixed(trainer.predict(x_test_adv), cifar_label_test[:100])

    x_test_adv_pred_after_attack = np.argmax(trainer.predict(x_test_adv), axis=1)
    nb_correct_adv_pred = np.sum(x_test_adv_pred_after_attack == np.argmax(cifar_label_test[:100], axis=1))

    print(string)
    print("Correctly classified: {}".format(nb_correct_adv_pred))
    print("Incorrectly classified: {}".format(100 - nb_correct_adv_pred))

    print(ex / 10000)
    return ex / 10000

def adverserial_predict(load_weight, classifier_weight):
    cifar_train, cifar_label_train, cifar_test, cifar_label_test = load_dataset_cifar()

    trainer, classifier = adverserial_classifier_setup(load_weight, classifier_weight)

    attacker = FastGradientMethod(classifier, eps=0.5)
    x_test_adv = attacker.generate(cifar_test[:100])

    predict_in_general(trainer, "", classifier)

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

def adverserial_layerwise_predict(load_weight, classifier_weight, layer_number):
    trainer, classifier = adverserial_classifier_setup(load_weight, classifier_weight)
    string_orig = "Adverserial Training: Adverserial Test: Original"
    # predict_in_general(trainer, string_orig, classifier)

    # Here is the command we had used for the Adversarial Training
    attacks = BasicIterativeMethod(classifier, eps=0.3, eps_step=0.01, max_iter=1000)
    trainer_new = MyAdvTrainer(classifier, attacks, ratio=1.0)

    model = trainer._classifier._model
    print(model.summary())
    print(len(model.layers))

    model_edited = model_layer_weight_zero(layer_number, model)

    comms_round = 1
    lr = 0.01
    loss_object = tf.keras.losses.CategoricalCrossentropy()
    metrics = ['accuracy']
    optimizer = SGD(lr=lr,
                    decay=lr / comms_round,
                    momentum=0.9
                    )

    classifier_new = MyTfClassifier(
        model=model_edited,
        loss_object=loss_object,
        train_step=train_step,
        nb_classes=10,
        input_shape=(28, 28, 1),
        clip_values=(0, 1)
    )

    trainer_new._classifier = classifier_new

    string_new = f"hidden the {layer_number}"
    predict_in_general(trainer_new, string_new, classifier_new)







