from flask import *;
from preprocess import predict_e;
app = Flask(__name__)

###############################################################################
#                       SETTING UP APP ROUTES                                 #
###############################################################################


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/response", methods=["GET", "POST"])
def response():

    if request.method == "POST":
        snippet = request.form["fsnippet"]
        # Testing with predict.py
        print(snippet)
        personality_type = predict_e(snippet)
    return render_template("response.html", name=personality_type, string=snippet)

'''
@app.route("/analysis")
def analysis():
    return render_template("analysis.html")


@app.route("/methodology")
def methodology():
    return render_template("methodology.html")


@app.route("/about")
def about():
    return render_template("about.html")
'''

###############################################################################
#                                   MAIN                                      #
###############################################################################

if __name__ == "__main__":
    app.run(debug=True)
