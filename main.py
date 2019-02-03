from eightnet import eightNet
import server

NUM_IMAGES = 1000


eightNet.load_data(NUM_IMAGES)
eightNet.initialize_model()
server.start_server()