from webob import Response
from eightnet import EightNet


def index(request):
    with open("index.html", "r") as f:
        return Response(f.read())
    
def trainNet(request):
    EightNet.train_model(10)
    return Response("Successfully trained model")