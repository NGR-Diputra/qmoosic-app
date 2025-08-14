from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import os
from utils.predict import predict
from vercel_flask import Vercel

app = Flask(__name__)

# Folder upload di dalam static agar bisa diakses langsung
UPLOAD_FOLDER = os.path.join(app.root_path, 'static', 'uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 20 * 1024 * 1024  # 20MB max

# Pastikan folder uploads ada
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files.get('audio')
        if not file or file.filename == '':
            return redirect(request.url)

        if file and file.filename.lower().endswith('.mp3'):
            try:
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)

                pred_label = predict(filepath)

                return redirect(url_for('show_result',
                                        filename=filename,
                                        pred_label=pred_label))

            except Exception as e:
                return render_template('index.html',
                                       error=f"Terjadi kesalahan: {str(e)}")

    return render_template('index.html')

@app.route('/result')
def show_result():
    filename = request.args.get('filename')
    pred_label = int(request.args.get('pred_label'))

    mood_map = {
        0: 'Kuadran I (High Arousal & Positive Valence)',
        1: 'Kuadran II (High Arousal & Negative Valence)',
        2: 'Kuadran III (Low Arousal & Negative Valence)',
        3: 'Kuadran IV (Low Arousal & Positive Valence)',
    }
    prediction = mood_map.get(pred_label, 'Tidak diketahui')

    image_filename = f"Q{pred_label + 1}.png"

    return render_template('result.html',
                           prediction=prediction,
                           pred_label=pred_label,
                           audio_filename=filename,
                           quadrant_image=image_filename)

# Handler untuk Vercel
app = Vercel(app)

if __name__ == '__main__':
    app.run(debug=True)
