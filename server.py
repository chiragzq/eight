from wsgiref.simple_server import make_server
from simplerouter import Router

def start_server():
    print("Starting server at port 8080")
    router = Router()

    router.add_route("/", "views:index")
    router.add_route("/trainNet", "views:trainNet")
    router.add_route("/trainProgress", "views:getTrainingProgress")
    router.add_route("/predict/{image}", "views:predict")

    application = router.as_wsgi

    make_server('', 8080, application).serve_forever()