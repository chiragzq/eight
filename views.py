from webob import Response
from eightnet import eightNet
import _thread
import json

def render(path):
    with open(path, "r") as f:
        return Response(f.read())

def index(request):
    return render("index.html")
    
def trainNet(request):
    _thread.start_new_thread(eightNet.train_model, (10,))
    return Response("Training model")

def getTrainingProgress(request):
    try:
        (p, q) = eightNet.get_training_progress()
    except AttributeError:
        (p, q) = (0, 100)
    return Response("%d %d" % (p, q))

def predict(request):
    return Response(str(eightNet.get_number([json.loads(request.urlvars["image"])])))
    
def test(request):
    return Response(eightNet.test())
    
def save(request):
    eightNet.save_model();
    with open("models/model.h5", "rb") as f:
        data = f.read()
        return Response(body=data, status=200, content_type="application/x-hdf")
        
def load(request):
    return render("load.html")
    
def get_models(request):
    return Response(",".join(eightNet.get_models()))
    
def load_model(request):
    name = request.urlvars["name"]
    eightNet.load_model(name)