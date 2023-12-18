from flask import Flask, render_template, request, redirect, url_for, send_from_directory
import os
from werkzeug.utils import secure_filename
from Detector import Detector

app = Flask(__name__)
detector = Detector(use_cuda=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    process_name = request.form.get('process_name', '')

    if not process_name:
        return redirect(url_for('index'))

    if 'file' not in request.files:
        return redirect(url_for('index'))

    file = request.files['file']

    if file.filename == '':
        return redirect(url_for('index'))

    if file:
        # İşlem adına göre ana klasör oluştur
        process_folder = os.path.join('uploads', process_name)
        os.makedirs(process_folder, exist_ok=True)

        # İşlem adına göre yükleme klasörü oluştur
        upload_folder = os.path.join(process_folder, 'uploads')
        os.makedirs(upload_folder, exist_ok=True)

        filename = secure_filename(file.filename)
        file_path = os.path.join(upload_folder, filename)
        file.save(file_path)

        if file_path.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
            # Sadece desteklenen görüntü türleri için processImage'i çağır
            detector.processImage(file_path)

            # Yüzleri kaydet ve dosya adlarını al
            faces = detector.saveFaces(os.path.join(process_folder, 'output'))
            faces_count = len(faces)
            face_images = [face for face in faces]

        else:
            # Videoyu işle ve yüzleri kaydet
            detector.processVideo(file_path)

            # Yüzleri kaydet ve dosya adlarını al
            faces = detector.saveFaces(os.path.join(process_folder, 'output'))
            faces_count = len(faces)
            face_images = [face for face in faces]

        return render_template('result.html', faces_count=faces_count, face_images=face_images, process_name=process_name)

@app.route('/getOutput/<process_name>/<path:file_name_output>')
def getOutput(process_name, file_name_output):
    # Temizleme işlemi: uploads\process_name\output\ kısmını çıkar
    base_path = os.path.join('uploads', process_name, 'output')
    cleaned_path = os.path.relpath(file_name_output, base_path)
    return send_from_directory(base_path, cleaned_path)

if __name__ == '__main__':
    if not os.path.exists('uploads'):
        os.makedirs('uploads')

    # IP: 0.0.0.0, Port: 5000
    app.run(host='0.0.0.0', port=5000, debug=True)
