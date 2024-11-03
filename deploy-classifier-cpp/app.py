from flask import Flask, render_template, request, jsonify
import subprocess
from gevent.pywsgi import WSGIServer

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/run', methods=['POST'])
def run_executable():
    # Check if files were uploaded
    image_file = request.files.get('image')
    model_file = request.files.get('model')
    model_xml_file = request.files.get('modelXml')
    model_bin_file = request.files.get('modelBin')
    runtime = request.form.get('runtime', 'opencv')  # Default to 'opencv' if not selected

    if runtime == 'opencv':
        exe = './Infer_OpenCV.exe'
    if runtime == 'ort':
        exe = './Infer_ORT.exe'
    if runtime == 'openvino':
        exe = './Infer_OpenVINO.exe'

    if runtime == 'opencv' or runtime == 'ort':
        DEFAULT_MODEL_PATH = "./models/vgg.onnx"
    if runtime == 'openvino':
        DEFAULT_MODEL_PATH = "./models/openvino/vgg.xml"

    if not image_file:
        return jsonify({'error': 'Image file is required'}), 400
    
    # Save the uploaded files
    image_path = './uploads/' + image_file.filename
    image_file.save(image_path)

    # Use the uploaded model or fall back to the default model
    if runtime == 'opencv' or runtime == 'ort':
        if model_file:
            model_path = './uploads/' + model_file.filename
            model_file.save(model_path)
        else:
            model_path = DEFAULT_MODEL_PATH
    
    if runtime == 'openvino':
        if model_xml_file and model_bin_file:
            model_xml_path = './uploads/' + model_xml_file.filename
            model_bin_path = './uploads/' + model_bin_file.filename
            model_xml_file.save(model_xml_path)
            model_bin_file.save(model_bin_path)
            model_path = model_xml_path
        else:
            model_path = DEFAULT_MODEL_PATH

    try:
        # Run the executable with the input arguments
        result = subprocess.run([exe, model_path, image_path], capture_output=True, text=True, timeout=10)

        # Check if the command executed successfully
        if result.returncode == 0:
            return jsonify({'output': result.stdout})
        else:
            return jsonify({'error': result.stderr})
    
    except subprocess.TimeoutExpired:
        return jsonify({'error': 'The executable took too long to respond and timed out.'}), 504  # HTTP 504 for timeout

    except Exception as e:
        return jsonify({'error': str(e)}), 500 # HTTP 500 for other errors

if __name__ == '__main__':
    # Debug/Development
    # app.run(debug=True, host="0.0.0.0", port=5000)
    # Production
    http_server = WSGIServer(('', 5000), app)
    http_server.serve_forever()

