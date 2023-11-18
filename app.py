import os
from move_comparison import compare_positions
from decimal import Decimal

from flask import (Flask, redirect, render_template, request,jsonify, make_response,
                   send_from_directory, url_for)

app = Flask(__name__)


@app.route('/')
def index():
   print('Request for index page received')
   return render_template('index.html')

@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'),
                               'favicon.ico', mimetype='image/vnd.microsoft.icon')

@app.route('/hello', methods=['POST'])
def hello():
   name = request.form.get('name')

   if name:
       print('Request for hello page received with name=%s' % name)
       return render_template('hello.html', name = name)
   else:
       print('Request for hello page received with no name or blank name -- redirecting')
       return redirect(url_for('index'))

@app.route('/posechecksat', methods=['GET'])
def posechecksat():
    inputbenchmarkfile = request.args.get('inputbenchmarkfile')
    inputbenchmarkblobcontainer = request.args.get('inputbenchmarkblobcontainer')
    inputplayerblobcontainer = request.args.get('inputplayerblobcontainer')
    inputplayerfile = request.args.get('inputplayerfile')
    outputblobfilename = request.args.get('outputblobfilename')
    outputblobfullfilename = request.args.get('outputblobfullfilename')
    outputblobcontainer = request.args.get('outputblobcontainer')
    checkrate =  Decimal(request.args.get('checkrate'))
    blobconn = request.args.get('blobconn')
    deletedblob= request.args.get('deletedblob')
    sport = request.args.get('sport')

#    blobcheckname = 'https://sportatousblob.blob.core.windows.net/'

    ## below this we need to commnent them out

#    blobconn='DefaultEndpointsProtocol=https;AccountName=sportatousblob;AccountKey=IyFEVlARtfoLlOQzQs/yapX3LIzMuK9rsIfWlojDiOzPmyEfHwgk+hXcVPYWwOie61pypEXcq5Ip+AStoEwOng==;EndpointSuffix=core.windows.net'
#    inputbenchmarkfile = 'pla/1/video/mati/matplavideo_mat_videofni1_1.mp4'
#    inputplayerfile = 'pla/7/video/mdti/mdtvideo_mdt_videofni7_5202307091526.mp4' # replace with 0 for webcam
#    outputblobfilename = 'out_mdtvideo_mdt_videofni7_5202307091526.mp4'
#    outputblobfullfilename='pla/7/video/mdto/out_mdtvideo_mdt_videofni7_5202307091526.mp4'
#    outputblobcontainer='satprdfalse'
#    inputbenchmarkblobcontainer='satprdfalse'
#    inputplayerblobcontainer='satprdfalse'
#    checkrate=60
#    sport='abc'

    try:
        if (blobconn.startswith('DefaultEndpointsProtocol=https;AccountName=sportatousblob;') or blobconn == ''):
#            print('Request for ai pose comparison')
            returncode = compare_positions(inputbenchmarkfile, inputplayerfile, inputbenchmarkblobcontainer, inputplayerblobcontainer, outputblobfilename ,outputblobfullfilename,outputblobcontainer,checkrate,blobconn,sport, False, True, deletedblob)
            return make_response(jsonify(returncode), 200)
        else:
            return  make_response(jsonify(-1), 400)
    except Exception as error:
        print("An exception occurred:", error) # An exception occurred: division by zero
        return  make_response(jsonify(-1), 400)
    
if __name__ == '__main__':
   app.run()
