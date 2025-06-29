from flask import Flask, request, render_template
import os, cv2, numpy as np, base64
from tensorflow.keras.models import load_model

app = Flask(__name__)
model = load_model(r"C:\Users\Mounica\Desktop\hemotovision\model\Blood Cell.h5")
CLASS_LABELS = ['BASOPHIL', 'EOSINOPHIL', 'LYMPHOCYTE', 'MONOCYTE', 'NEUTROPHIL']

def predict_image(img_path):
    img = cv2.imread(img_path)
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (224, 224))
    x = np.expand_dims(resized / 255.0, axis=0)
    pred = model.predict(x)[0]
    return CLASS_LABELS[np.argmax(pred)], rgb

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        f = request.files.get('file')
        if f:
            save_path = os.path.join('static', f.filename)
            f.save(save_path)
            label, img_rgb = predict_image(save_path)

           _, img_enc = cv2.imencode('.png', cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
            img_b64 = base64.b64encode(img_enc).decode('utf-8')
            return render_template('result.html', class_label=label, img_data=img_b64)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)

