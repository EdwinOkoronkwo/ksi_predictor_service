from flask import Flask
from flask_cors import CORS
import logging
from backend.routes.predict import predict_bp
from backend.routes.feature_info import feature_bp

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
sys.path.append(str(ROOT_DIR))

# Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

app = Flask(__name__)
CORS(app)
app.register_blueprint(predict_bp)
app.register_blueprint(feature_bp)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)