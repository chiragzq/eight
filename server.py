from wsgiref.simple_server import make_server
from simplerouter import Router

def start_server():
    print("Starting server at port 8080")
    router = Router()

    router.add_route("/", "views:index")
    router.add_route("/index.html", "views:index")
    router.add_route("/trainNet", "views:trainNet")
    router.add_route("/trainProgress", "views:getTrainingProgress")
    router.add_route("/predict/{image}", "views:predict")
    router.add_route("/test", "views:test")
    router.add_route("/save", "views:save")
    router.add_route("/load", "views:load")
    router.add_route("/models", "views:get_models")
    router.add_route("/loadModel/{name}", "views:load_model")

    application = router.as_wsgi

    make_server('', 8080, application).serve_forever()