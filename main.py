from customized_model.adverserial_model import adverserial_train
from customized_model.adverserial_predict import *

def adverserial_train_full_pipeline():
    save_weight = "adversarial-robustness-toolbox/weight_adverserial_training/trained_epoch_24_max_iter_1000.h5"
    load_weight = "adversarial-robustness-toolbox/weight_adverserial_training/trained_epoch_30_max_iter_100.h5"
    classifier_weight = "adversarial-robustness-toolbox/weights_preliminary_classifier/trained_100_epochs.h5"
    iter_no = 1000
    epoch_no = 3
    layer_number = 8
    for i in range(100):
        layer_number = i
        adverserial_layerwise_predict(load_weight, classifier_weight, layer_number)

    # adverserial_train(save_weight, load_weight, classifier_weight, iter_no, epoch_no)

    # adverserial_predict(load_weight, classifier_weight)
if __name__ == '__main__':
    adverserial_train_full_pipeline()
