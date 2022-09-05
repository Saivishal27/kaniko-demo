#!flask/bin/python
from flask import Flask, jsonify,request
import multiprocessing
#from model_wrapper import Model
import pickle
import pandas as pd
import os
#remember to install sys and time inside each Docker Containers
import sys
import time
from ctypes import c_char_p
# manager = multiprocessing.Manager()


app = Flask(__name__)

# base_path = '/volume_mapping'
base_path = 'D:\\SMAV\\model_files\\registered_models\\DG2_LUB_SYS_FAILURE\\ver_1\\model_files'
# model = None 

# Message = manager.Value(c_char_p, "Model Training Not Done Yet")
if __name__ == '__main__':
    manager = multiprocessing.Manager()
    Message = manager.Value(c_char_p, "Model Training Not Done Yet")
    
locked=False

def sub_process(model,data,model_filename):
    global base_path
    Message.value = 'Model Training is in process'
    try:
        model.train(data)
        pickle.dump(model,open(model_filename, 'wb'))
    except Exception as e:
        time.sleep(15)
        Message.value = str(e)
    return True

@app.route('/gateway', methods=['POST','GET'])
def gateway():
    global locked
    global base_path
    model_filename = 'model.pkl'
    input = request.json
    request_type = input['request_type']
    if input['data_file_name'] != None:
        data = pd.read_csv('/volume_mapping/'+input['data_file_name'])
        # data = pd.read_csv(input['data_file_name'])
    elif input['data']!=None:
        data = pd.read_json (input['data']).reset_index(drop=True)
    
    if request_type == 'train':
        try:
            use_cols    = input['use_cols']
            labels      = input['labels']
            window_size = input['window_size']
            stride      = input['stride']
            while locked:
                time.sleep(1)
            locked = True
            from model_wrapper import Model
            locked = False
            model = Model(use_cols,labels,window_size,stride)
            thread = multiprocessing.Process(target=sub_process, args=(model,data,model_filename,))
            thread.start()

            response = {
                    'train_report'          : None,
                    'test_report'           : None,
                    'hyper_parm_list'      : None
            }
            return jsonify({'training_started':True,
                    'message': "Training is Started Successfully",
                    'status': 200})
        except Exception as e:
            return jsonify({'training_started':False,
                    'message': "Training couldn't able to start. Exception occured: "+str(e),
                    'status': 200})
        
    elif request_type == 'predict':   
        try:
            if input['model_type'] == 'ML':
                if os.path.exists(model_filename):
                    while locked:
                        time.sleep(1)
                    locked = True
                    from model_wrapper import Model
                    model = pickle.load(open(model_filename, 'rb'))
                    del sys.modules['model_wrapper']
                    locked = False
                    predictions = model.predict(data,input['target_label'])
                    
                    response_data = {'predictions':predictions.to_json(),
                                        'train_report': model._train_report,
                                        'test_report': model._test_report,
                                        'hyper_parm_list': model._hyper_parm_list} 
                else:
                    return jsonify({'data':None, 'prediction_successful':False,'message': Message.value,
                            'status': 404}
                            )
            elif input['model_type'] == 'PATTERN':
                while locked:
                    time.sleep(0.1)
                locked = True
                from model_wrapper import Model
                model = Model(input['use_cols'],[],input['window_size'],input['stride'])
                del sys.modules['model_wrapper']
                locked=False
                predictions = model.predict(data,input['target_label'])
                response_data = {'predictions':predictions.to_json()} 

            return jsonify({'data':response_data, 'prediction_successful':True, 'message': "Predicted successfully",
                            'status': 200}
                            )
        except Exception as e:
            return jsonify({'data':'', 'prediction_successful':False,
                    'message': "Exception occured: "+str(e),
                    'status': 200})
    elif request_type == 'test':
        try:
            if input['model_type'] == 'ML':
                if os.path.exists(model_filename):
                    metrics = input['metrics']
                    model = pickle.load(open(model_filename, 'rb'))
                    scores = model.test(data,metrics)
                    response_data = {'current_test_scores' : scores,
                                        'train_report': model._train_report,
                                        'test_report': model._test_report,
                                        'hyper_parm_list': model._hyper_parm_list} 
                else:
                    return jsonify({'data':None, 'computed_scores':False,'message': Message.value,
                            'status': 404}
                            )
            elif input['model_type'] == 'PATTERN':
                raise Exception("PATTERN RECOGNITION MODELS CANNOT BE TESTED")
            return jsonify({'data':response_data, 'computed_scores':True, 'message': "Predicted successfully",
                            'status': 200}
                            )
        except Exception as e:
            return jsonify({'data':'', 'computed_scores':False,
                    'message': "Exception occured: "+str(e),
                    'status': 200})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080 , debug=True)
