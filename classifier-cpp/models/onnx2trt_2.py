from polygraphy.backend.trt import (
    CreateConfig,
    EngineFromNetwork,
    NetworkFromOnnxPath,
    save_engine
)

def main(onnx_file):
    build_engine = EngineFromNetwork(
        NetworkFromOnnxPath(onnx_file),
        config=CreateConfig(int8=False),
    )

    save_engine(build_engine, 'vgg.engine')
        

if __name__ == "__main__":
    onnx_file = "./vgg.onnx"
    main(onnx_file)