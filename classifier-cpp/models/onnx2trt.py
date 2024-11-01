import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def build_engine(onnx_file_path, engine_file_path):
    with trt.Builder(TRT_LOGGER) as builder, builder.create_builder_config() as config, builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)) as network, trt.OnnxParser(network, TRT_LOGGER) as parser:
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1GB

        builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))

        # Load the ONNX model
        with open(onnx_file_path, 'rb') as model:
            if not parser.parse(model.read()):
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
                return None

        # Build and serialize the engine
        serialized_engine = builder.build_serialized_network(network, config)

        print("===========", serialized_engine, network, config)

        with open(engine_file_path, 'wb') as f:
            f.write(serialized_engine)

def main():
    onnx_file_path = 'vgg.onnx'
    engine_file_path = 'vgg.engine'
    build_engine(onnx_file_path, engine_file_path)

if __name__ == '__main__':
    main()