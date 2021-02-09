import tensorflow_hub as hub
import tensorflow as tf
my_devices = tf.config.list_physical_devices(device_type='CPU')
tf.config.set_visible_devices([], 'GPU')
english_sentences = tf.constant(["dog", "Puppies are nice.", "I enjoy taking long walks along the beach with my dog."])

preprocessor = hub.KerasLayer(
    "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/2")
encoder = hub.KerasLayer(
    "https://tfhub.dev/google/universal-sentence-encoder-cmlm/en-base/1")

english_embeds = encoder(preprocessor(english_sentences))["default"]

print (english_embeds)
