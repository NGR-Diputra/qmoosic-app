from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import os
from utils.predict import predict  # Asumsikan modul ini sudah Anda buat

app = Flask(__name__)
UPLOAD_FOLDER = os.path.join(app.root_path, 'static', 'uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 20 * 1024 * 1024  # Maksimal 20MB

# Buat folder uploads jika belum ada
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Pastikan ada file yang diunggah
        file = request.files.get('audio')
        if not file or file.filename == '':
            return redirect(request.url)

        if file and file.filename.endswith('.mp3'):
            try:
                # Simpan file yang aman
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)

                # Lakukan prediksi dari file yang diunggah
                pred_label = predict(filepath)

                # Alihkan ke halaman hasil dengan mengirimkan data
                return redirect(url_for('show_result', filename=filename, pred_label=pred_label))

            except Exception as e:
                # Tangani kesalahan jika prediksi gagal
                return render_template('index.html', error=f"Terjadi kesalahan: {str(e)}")

    # Tampilkan halaman utama jika metode GET
    return render_template('index.html')

@app.route('/result')
def show_result():
    # Ambil data dari parameter URL
    filename = request.args.get('filename')
    pred_label = int(request.args.get('pred_label'))

    # Peta mood untuk penjelasan
    mood_map = {
        0: 'Kuadran I (High Arousal & Positive Valence)',
        1: 'Kuadran II (High Arousal & Negative Valence)',
        2: 'Kuadran III (Low Arousal & Negative Valence)',
        3: 'Kuadran IV (Low Arousal & Positive Valence)',
    }
    prediction = mood_map.get(pred_label, 'Tidak diketahui')

    # Tentukan nama file gambar berdasarkan label prediksi
    image_filename = f"Q{pred_label + 1}.png"

    return render_template(
        'result.html',
        prediction=prediction,
        pred_label=pred_label,
        audio_filename=filename,
        quadrant_image=image_filename
    )

if __name__ == '__main__':
    app.run(debug=True)