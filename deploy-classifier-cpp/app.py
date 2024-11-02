from flask import Flask, render_template, request, jsonify
import subprocess
from gevent.pywsgi import WSGIServer

app = Flask(__name__)

DEFAULT_MODEL_PATH = "./models/vgg.onnx"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/run', methods=['POST'])
def run_executable():
    # Check if files were uploaded
    image_file = request.files.get('image')
    model_file = request.files.get('model')
    runtime = request.form.get('runtime', 'opencv')  # Default to 'cpu' if not selected

    if runtime == 'opencv':
        exe = './Infer_OpenCV.exe'
    if runtime == 'ort':
        exe = './Infer_ORT.exe'

    if not image_file:
        return jsonify({'error': 'Image file is required'}), 400
    
    # Save the uploaded files
    image_path = './uploads/' + image_file.filename
    image_file.save(image_path)

    # Use the uploaded model or fall back to the default model
    if model_file:
        model_path = './uploads/' + model_file.filename
        model_file.save(model_path)
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

