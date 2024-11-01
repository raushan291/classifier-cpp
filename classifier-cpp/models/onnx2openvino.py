import os

onnx_path = "./vgg.onnx"
openvino_dir = "./openvino"

mo_command = f"""mo
                 --input_model "{onnx_path}"
                 --output_dir "{openvino_dir}"
                 --input_shape "{[1, 3, 224, 224]}"
                 """
# --compress_to_fp16

mo_command = " ".join(mo_command.split())

print("Model Optimizer command to convert the ONNX model to OpenVINO:\n", mo_command)

os.system(mo_command)

print("openvino conversion done..")
