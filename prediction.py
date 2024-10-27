 
import joblib
import numpy
def predict(input):
     input=numpy.array([input])
     loaded_model=joblib.load('modelpipe.joblib')
     out=loaded_model.predict(input)
     if out==1:
        return "Spam"
     else:
       return "Not Spam"
    
