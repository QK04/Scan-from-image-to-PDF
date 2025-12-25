from flask import Flask, render_template, request, send_file, jsonify
from scanner import scan_document, manual_warp
import img2pdf, io, base64

app = Flask(__name__)

@app.route('/')
def index(): return render_template('index.html')

@app.route('/process_auto', methods=['POST'])
def process_auto():
    file = request.files.get('image')
    scanned_data = scan_document(file.read())
    return jsonify({'image': base64.b64encode(scanned_data).decode('utf-8')})

@app.route('/process_manual', methods=['POST'])
def process_manual():
    data = request.json
    img_bytes = base64.b64decode(data['image'])
    warped_data = manual_warp(img_bytes, data['points'])
    return jsonify({'image': base64.b64encode(warped_data).decode('utf-8')})

@app.route('/convert', methods=['POST'])
def convert():
    data = request.json.get('images', [])
    img_list = [base64.b64decode(i) for i in data]
    return send_file(io.BytesIO(img2pdf.convert(img_list)), 
                     mimetype='application/pdf', as_attachment=True, 
                     download_name='Document_Scan.pdf')

if __name__ == '__main__':
    app.run(debug=True)