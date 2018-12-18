from webob import Response
from eightnet import EightNet


def index(request):
    return Response("""
<!DOCTYPE html>
<html>
    <head>
        <title>EightNet</title>
        <script src="https://code.jquery.com/jquery-3.3.1.min.js" integrity="sha256-FgpCb/KJQlLNfOu91ta32o/NMZxltwRo8QtmkMRdAu8=" crossorigin="anonymous"></script>
    </head>
    <body>
        <p> Welcome to EightNet! </p>
        <button id="trainNet"> Train Net </button>
        <p id="result"></p>
        <script>
            $("#trainNet").click((e) => {
                e.preventDefault();
                $.ajax({
                    url: "/trainNet",
                    type: "get",
                    beforeSend: () => {
                        $("#result").html("<b>Loading...</b>");
                    },
                    success: (data, status) => {
                        $("#result").html(data);
                    }
                });
            });
        </script>
    </body>
</html>
    """)
    
def trainNet(request):
    EightNet.train_model(10)
    return Response("Successfully trained model")