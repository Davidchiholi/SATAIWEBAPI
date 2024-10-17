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
    model = request.args.get('model')    
    equip = request.args.get('equip')
    model1 = request.args.get('model1')    
    equip1 = request.args.get('equip1')
    model2 = request.args.get('model2')    
    equip2 = request.args.get('equip2')
    inputbenchmarkfile = request.args.get('inputbenchmarkfile')
    inputbenchmarkblobcontainer = request.args.get('inputbenchmarkblobcontainer')
    inputplayerblobcontainer = request.args.get('inputplayerblobcontainer')
    inputplayerfile = request.args.get('inputplayerfile')
    outputblobfilename = request.args.get('outputblobfilename')
    outputblobfullfilename = request.args.get('outputblobfullfilename')
    outputblobcontainer = request.args.get('outputblobcontainer')
    checkrate =  request.args.get('checkrate')
    blobconn = request.args.get('blobconn')
    deletedblob= request.args.get('deletedblob')
    sport = request.args.get('sport')
    joints = request.args.get('joints')
    if model == 'NA':
        model = ''
    if model1 == 'NA':
        model1 = ''
    if model2 == 'NA':
        model2 = ''
    checkrate_in_decimal = Decimal(checkrate)
#    checkrate = 0.1

#    blobcheckname = 'https://sportatousblob.blob.core.windows.net/'

    ## below this we need to commnent them out

 #   inputbenchmarkfile = 'pla/1/video/pvm/pvmvideo_pvm_videofn_golf_1.mp4'
 #   inputplayerfile = 'pla/1/video/pvmi/pvmvio_pvm_videofniuid_1_golf_1.mp4' # replace with 0 for webcam
 #   outputblobfilename = 'out_pvmvio_pvm_videofniuid_1_golf_1.mp4'
 #   outputblobpath = 'pla/1/video/pvmo/'  
 #   outputblobfullfilename='pla/1/video/pvmo/out_pvmvio_pvm_videofniuid_1_golf_1.mp4'
 #   outputblobcontainer='satprdfalse'
 #   inputbenchmarkblobcontainer='satprdfalse'
 #   inputplayerblobcontainer='satprdfalse'
 #   checkrate=0.150000
 #   sport='GOLF'
 #   model=''
 #   model1=''
 #   model2=''
 #   equip='NA'
 #   equip1='NA'
 #   equip2='NA'
    joints_dict = {}
    joints_list = joints.split(",")
    joints_dict["joint1"] = int(joints_list[0])
    joints_dict["joint1_weighting"] = float(joints_list[1])
    joints_dict["joint2"] = int(joints_list[2])
    joints_dict["joint2_weighting"] = float(joints_list[3])
    joints_dict["joint3"] = int(joints_list[4])
    joints_dict["joint3_weighting"] = float(joints_list[5])
    joints_dict["joint4"] = int(joints_list[6])
    joints_dict["joint4_weighting"] = float(joints_list[7])          
    
    try:
        if (blobconn.startswith('DefaultEndpointsProtocol=https;AccountName=sportatousblob;') or blobconn == ''):
#            print('Request for ai pose comparison')
            returncode = compare_positions(inputbenchmarkfile, inputplayerfile, inputbenchmarkblobcontainer, inputplayerblobcontainer, outputblobfilename ,outputblobfullfilename,outputblobcontainer,checkrate_in_decimal,blobconn,sport, False, True, deletedblob, model, equip, model1, equip1, model2, equip2, joints_dict)
            return make_response(jsonify(returncode), 200)
        else:
            return  make_response(jsonify(-1), 400)
    except Exception as error:
        print("An exception occurred:", error) # An exception occurred: division by zero
        return  make_response(jsonify(-1), 400)
    
if __name__ == '__main__':
   app.run()
