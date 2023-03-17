import json
import os
import shutil
import time
import urllib
import uuid
import logging

import config
from config import MEDIA_ROOT, PREDICTOR, cfg, segmentation_module, use_second_gpu
from flask import request  # jsonify
from flask_cors import cross_origin
from flask_restful import Api, Resource
from inference import run
from material_segmentation.run import material_segmentation
from PIL import Image

# from tasks import celery

api = Api(prefix=config.API_PREFIX)

logger = logging.getLogger()

# class TaskStatusAPI(Resource):
#    def get(self, task_id):
#        task = celery.AsyncResult(task_id)
#        return jsonify(task.result)

# class DataProcessingAPI(Resource):
#    def post(self):
#        task = process_data.delay()
#        return {'task_id': task.id}, 200


# @celery.task()
# def process_data():
#    time.sleep(60)

class SyncProcessAPIView(Resource):
   """POST API class"""
   @cross_origin()
   def post(self):
      """
      (POST)
      upload: <urlstr/image>
      returns json data from image
      """
      start_time = time.time()
      logger.info('new img')
      res = {
         "results": [],
         "errors": {},
         "success": False
      }
      data = request.form
      try:
         upload = data["upload"]
         path = os.path.join(MEDIA_ROOT, str(uuid.uuid4()) + ".jpg")
         try:
               urllib.request.urlretrieve(upload, path)
         except:
               res["errors"]["sync"] = f"001: Could not load image from url: {upload}"
               return res
      except:
         upload = request.files["upload"]
         path = os.path.join(MEDIA_ROOT, str(uuid.uuid4()) + ".jpg")
         upload.save(path)

      try:
         Image.open(path)
      except:
         res["errors"]["sync"] = "002: Invalid img file"
         return res

      objs, image_height, image_width = run(path, PREDICTOR)
      res["results"].extend(objs)
      seg_results, seg_url = material_segmentation(path, cfg, segmentation_module, use_second_gpu)
      res["results"].extend(seg_results)
      res["segments_url"] = seg_url
      res["img_attributes"] = {
         "height": image_height,
         "width": image_width
      }
      res["image"] = path.split('/')[-1]

      #archive input and response
      archive_basepath = f'archive/{str(uuid.uuid4())}'
      shutil.copyfile(path, archive_basepath + '.png')
      with open(archive_basepath + '.json', "w") as file:
         file.write(json.dumps(res, indent=4))
      #delete original
      os.remove(path)

      processing_time = time.time() - start_time
      res["processing_time"] = processing_time
      res["success"] = True

      return res

# data processing endpoint
#api.add_resource(DataProcessingAPI, '/process_data')
api.add_resource(SyncProcessAPIView, '/v1/furniture')

# task status endpoint
# api.add_resource(TaskStatusAPI, '/tasks/<string:task_id>')