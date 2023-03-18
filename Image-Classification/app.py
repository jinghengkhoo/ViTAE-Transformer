import logging
import os
import urllib
import uuid
import shutil
from PIL import Image
from argparse import Namespace

from inference import load_model, run
from flask import Flask, request
from flask_cors import CORS, cross_origin
from flask_restful import Api, Resource

project_dir = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__)
logging.basicConfig(level=logging.DEBUG,
                    format='[%(asctime)s]: {} %(levelname)s %(message)s'.format(os.getpid()),
                    datefmt='%Y-%m-%d %H:%M:%S',
                    handlers=[logging.StreamHandler()])
cors = CORS(app, resources={r"/*": {"origins": "*"}})
app.config.update(
    CORS_HEADERS='Content-Type'
)

logger = logging.getLogger()

args = Namespace(amp=False,
                apex_amp=False,
                batch_size=64,
                bn_eps=None,
                bn_momentum=None,
                bn_tf=False,
                channels_last=False,
                clip_grad=None,
                crop_pct=None,
                decay_rate=0.1,
                device='cuda:0',
                dist_bn='',
                distributed=False,
                drop=0.0,
                drop_block=None,
                drop_connect=None,
                drop_path=0.1,
                eval_checkpoint='/workspace/Image-Classification/ViTAE-T.pth.tar',
                eval_metric='top1',
                gp=None,
                img_size=224,
                interpolation='',
                local_rank=0,
                log_interval=50,
                mean=None,
                model='ViTAE_basic_Tiny',
                model_ema=True,
                model_ema_decay=1.0,
                model_ema_force_cpu=False,
                momentum=0.9,
                native_amp=False,
                no_prefetcher=False,
                num_classes=1000,
                num_gpu=1,
                opt='adamw',
                opt_betas=None,
                opt_eps=None,
                output='',
                pin_mem=False,
                prefetcher=True,
                pretrained=False,
                rank=0,
                real_labels='./images/real.json',
                recovery_interval=0,
                results_file='',
                save_images=False,
                seed=42,
                smoothing=0.1,
                split_bn=False,
                std=None,
                sync_bn=False,
                train_interpolation='random',
                use_multi_epochs_loader=False,
                validation_batch_size_multiplier=1,
                weight_decay=0.05,
                workers=8,
                world_size=1)

model = load_model(args)

api = Api(app, prefix='/api')

@app.route("/")
def home():
    return ":D"

class RunInferenceAPIView(Resource):
    """POST API class"""
    @cross_origin()
    def post(self):
        res = {
            "results": {},
            "errors": {},
            "success": False
        }
        data = request.form
        foldername = str(uuid.uuid4()) + ".png"
        folder_path = os.path.join("/workspace/Image-Classification", foldername)
        try:
            upload = data["upload"]
            filename = urllib.parse.urlparse(upload).path.split("/")[-1]
            if not os.path.splitext(filename)[1]:
                filename += ".png"
            path = os.path.join(folder_path, filename)
            try:
                os.makedirs(folder_path)
                urllib.request.urlretrieve(upload, path)

            except:
                res["errors"]["run"] = "001: Could not load image from url: {}".format(upload)
                return res
        except:
            upload = request.files["upload"]
            path = os.path.join(folder_path, upload.filename)
            os.makedirs(folder_path)
            upload.save(path)

        try:
            Image.open(path)
        except:
            res["errors"]["run"] = "002: Invalid img file"
            return res

        res["results"] = run(model, args, folder_path)
        shutil.rmtree(folder_path)

        res["success"] = True
        return res


api.add_resource(RunInferenceAPIView, '/run')

if __name__ == "__main__":
    app.run(host='0.0.0.0',port=8000, debug=True)