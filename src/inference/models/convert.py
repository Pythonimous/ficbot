import tensorflow as tf
import os

#export WRAPT_DISABLE_EXTENSIONS=true

model = tf.keras.models.load_model('img2name.keras')
tf.saved_model.save(model, 'img2name')

converter = tf.lite.TFLiteConverter.from_saved_model('img2name')

converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS] 

tflite_model = converter.convert()

# Save the model.
with open('model.tflite', 'wb') as f:
  f.write(tflite_model)
