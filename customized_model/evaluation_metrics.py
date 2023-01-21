import numpy as np
import tensorflow as tf

def perplexity_medium_fixed(predictions, test):
  perplexity = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
  for i, pred in enumerate(list(predictions)):
    if np.argmax(pred) == np.argmax(test[i]):
      perplexity[np.argmax(pred)]=perplexity[np.argmax(pred)] + np.log2(pred[np.argmax(pred)])*100
  exposure = - perplexity
  return exposure

def model_layer_weight_zero(layer_number, model):

    model_copy= tf.keras.models.clone_model(model)
    print(model_copy.layers[layer_number])
    model_copy.weights[layer_number].assign(abs(model_copy.weights[layer_number])*0)
    return model_copy