from eightnet import EightNet
import server

NUM_IMAGES = 5000

EightNet.load_data(NUM_IMAGES)
EightNet.initialize_model()

server.start_server()