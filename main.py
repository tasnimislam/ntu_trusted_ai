from customized_model.adverserial_model import adverserial_train
from customized_model.adverserial_model import adverserial_predict
def adverserial_train_full_pipeline():
    save_weight = "adversarial-robustness-toolbox/weight_adverserial_training/trained_epoch_18_max_iter_1000.h5"
    load_weight = "adversarial-robustness-toolbox/weight_adverserial_training/trained_epoch_15_max_iter_1000.h5"
    classifier_weight = "adversarial-robustness-toolbox/weights_preliminary_classifier/trained_100_epochs.h5"
    iter_no = 1000
    epoch_no = 3
    # adverserial_train(save_weight, load_weight, classifier_weight, iter_no, epoch_no)
    adverserial_predict(load_weight, classifier_weight)
if __name__ == '__main__':
    adverserial_train_full_pipeline()
