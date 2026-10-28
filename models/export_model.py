import sys
from pathlib import Path


OUTPUT = Path(__file__).parent / "facenet512.onnx"

if OUTPUT.exists():
    sys.exit("The model already exists!")


import tensorflow as tf
import tf2onnx
from deepface import DeepFace

try:
    from deepface.models.facial_recognition.Facenet import FaceNet512dClient
    model = FaceNet512dClient().model
except ImportError:
    wrapper = DeepFace.build_model("Facenet512")
    model = getattr(wrapper, "model", wrapper)

spec = [tf.TensorSpec(model.inputs[0].shape, tf.float32, name="input_1")]
onnx_model, _ = tf2onnx.convert.from_keras(model, input_signature=spec, opset=13)

OUTPUT.write_bytes(onnx_model.SerializeToString())

print("The model is done!")
