class BaseConfig():
   API_PREFIX = '/api'
   TESTING = False
   DEBUG = False

class DevConfig(BaseConfig):
   FLASK_ENV = 'development'
   DEBUG = True