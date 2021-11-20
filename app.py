import cv2
from flask import Flask, app,jsonify,request,send_from_directory,render_template
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import numpy as np
import os

app=Flask(__name__)

STATIC_FOLDER='D:/PredCancer/static'

UPLOAD_FOLDER=STATIC_FOLDER+'/uploads' #path to the folder where we will store the upload before prediction
MODEL_FOLDER=STATIC_FOLDER+'/models' #path to the folders where we'll store the models

def predictBREAST(fullpath):
    data=image.load_img(fullpath,target_size=(75,75,3))
    data=np.expand_dims(data,axis=0)
    data = data.astype('float') / 255

    model=load_model(MODEL_FOLDER+'/BCModel.h5')
    result=model.predict(data)

    pred_prob=model.predict_proba(data)

    return result,pred_prob

def predictSKIN(fullpath):
    data=image.load_img(fullpath,target_size=(224,224,3))
    data=np.expand_dims(data,axis=0)
    data = data.astype('float') / 255

    model=load_model(MODEL_FOLDER+'/SCModel.h5')
    result=model.predict(data)

    pred_prob=model.predict_proba(data)

    return result,pred_prob

def predictLUNGS(fullpath):
    data=image.load_img(fullpath,target_size=(224,224,3))
    data=np.expand_dims(data,axis=0)
    data = data.astype('float') / 255

    model=load_model(MODEL_FOLDER+'/lung_custom_model.h5')
    result=model.predict(data)
    pred_prob=model.predict_proba(data)

  
    return result,pred_prob
def predictBRAIN(fullpath):
    data=image.load_img(fullpath,target_size=(150,150,3))
    data=np.expand_dims(data,axis=0)
    data = data.astype('float') / 255

    model=load_model(MODEL_FOLDER+'/custom_brain_model.h5')
    result=model.predict(data)
    pred_prob=model.predict_proba(data)

    return result,pred_prob



####################################################################################################


@app.route('/breast',methods=['GET','POST'])
def upload_file_breast():
    if request.method == 'GET':
        return render_template('index.html')
    else:
        file=request.files['image']
        fullname=os.path.join(UPLOAD_FOLDER,file.filename)
        file.save(fullname)

        result,pred_prob=predictBREAST(fullname)
        pred=np.argmax(result,axis=1)
        prob=np.max(pred_prob,axis=1)
        prob_str=' '.join(map(str, np.round_(prob*100,decimals=2)))

        if pred==0:
            label='IDC POSITIVE'
            accuracy=prob_str
            return render_template('prediction.html',image_file=file.filename,prediction_text='There is a {}% chance that the cells are {}.'.format(accuracy,label))

        elif pred==1:
            label='IDC NEGATIVE'
            accuracy=prob_str
            return render_template('prediction.html',image_file=file.filename,prediction_text='There is a {}% chance that the cells are {}.'.format(accuracy,label))


@app.route('/skin',methods=['GET','POST'])
def upload_file_skin():
    if request.method == 'GET':
        return render_template('index.html')
    else:
        file=request.files['image']
        fullname=os.path.join(UPLOAD_FOLDER,file.filename)
        file.save(fullname)

        result,pred_prob=predictSKIN(fullname)
        pred=np.argmax(result,axis=1)
        prob=np.max(pred_prob,axis=1)
        prob_str=' '.join(map(str, np.round_(prob*100,decimals=2)))

        if pred==0:
            label='Benign type of skin cancer'
            accuracy=prob_str
            return render_template('prediction.html',image_file=file.filename,prediction_text='The person has {}% of chance of having {}.'.format(accuracy,label))

        elif pred==1:
            label='Malignant type of skin cancer'
            accuracy=prob_str
            return render_template('prediction.html',image_file=file.filename,prediction_text='The person has {}% of chance of having {}.'.format(accuracy,label))

       

@app.route('/lungs',methods=['GET','POST'])
def upload_file_lungs():
    if request.method == 'GET':
        return render_template('index.html')
    else:
        file=request.files['image']
        fullname=os.path.join(UPLOAD_FOLDER,file.filename)
        file.save(fullname)

        result,pred_prob=predictLUNGS(fullname)
        pred=np.argmax(result,axis=1)
        prob=np.max(pred_prob,axis=1)
        prob_str=' '.join(map(str, np.round_(prob*100,decimals=2)))

        if pred==0:
            label='Adenocarcinoma'
            accuracy=prob_str
            return render_template('prediction.html',image_file=file.filename,prediction_text='These lungs have {}% chance of having {}.'.format(accuracy,label))

        elif pred==1:
            label='Large Cell Carcinoma'
            accuracy=prob_str
            return render_template('prediction.html',image_file=file.filename,prediction_text='These lungs have {}% chance of having {}.'.format(accuracy,label))

        elif pred==2:
            label='Healthy'
            accuracy=prob_str
            return render_template('prediction.html',image_file=file.filename,prediction_text='These lungs have {}% chance of having {}.'.format(accuracy,label))

        elif pred==3:
            label='Squamous Cell Carcinoma'
            accuracy=prob_str
            return render_template('prediction.html',image_file=file.filename,prediction_text='These lungs have {}% chance of having {}.'.format(accuracy,label))



@app.route('/brain',methods=['GET','POST'])
def upload_file_brain():
    if request.method == 'GET':
        return render_template('index.html')
    else:
        file=request.files['image']
        fullname=os.path.join(UPLOAD_FOLDER,file.filename)
        file.save(fullname)

        result,pred_prob=predictBRAIN(fullname)
        pred=np.argmax(result,axis=1)
        prob=np.max(pred_prob,axis=1)
        prob_str=' '.join(map(str, np.round_(prob*100,decimals=2)))


        if pred==0:
            label='Glioma Tumour'
            accuracy=prob_str
            return render_template('prediction.html',image_file=file.filename,prediction_text='This brain MRI have {}% chance of having {}.'.format(accuracy,label))

        elif pred==1:
            label='Meningioma Tumour'
            accuracy=prob_str
            return render_template('prediction.html',image_file=file.filename,prediction_text='This brain MRI have {}% chance of having {}.'.format(accuracy,label))

        elif pred==2:
            label='Healthy'
            accuracy=prob_str
            return render_template('prediction.html',image_file=file.filename,prediction_text='This brain MRI have {}% chance of having {}.'.format(accuracy,label))

        else:
            label='Pituitary Tumour'
            accuracy=prob_str
            return render_template('prediction.html',image_file=file.filename,prediction_text='This brain MRI have {}% chance of having {}.'.format(accuracy,label))


@app.route('/upload/<filename>')
def send_file(filename):
    return send_from_directory(UPLOAD_FOLDER,filename)

####################################################################################################

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/services')
def services():
    return render_template('services.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

if __name__=='__main__':
    app.run(debug=True)
        