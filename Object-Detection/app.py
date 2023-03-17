"""
Main module
"""

import logging
import os
import app_configs
from api import api

from flask import Flask
from flask_cors import CORS

app = Flask(__name__)
logging.basicConfig(level=logging.DEBUG,
                    filename=os.path.join(app.root_path, 'logs/flask.log'),
                    format='[%(asctime)s]: {} %(levelname)s %(message)s'.format(os.getpid()),
                    datefmt='%Y-%m-%d %H:%M:%S',
                    handlers=[logging.StreamHandler()])
cors = CORS(app, resources={r"/*": {"origins": "*"}})
app.config.update(
    CORS_HEADERS='Content-Type'
)
logger = logging.getLogger()

def create_app():
   logger.info(f'Starting app in {app_configs.APP_ENV} environment')
   app = Flask(__name__)
   app.config.from_object('config')
   api.init_app(app)
   # initialize SQLAlchemy
   #db.init_app(app)

   @app.route('/')
   def home():
       return ':D'
   return app


if __name__ == "__main__":
   app = create_app()
   app.run(host='0.0.0.0', port=8000, debug=True)