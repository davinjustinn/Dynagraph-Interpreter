import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
import pathlib
import pandas as pd
import cv2
import streamlit as st

from PIL import Image, ImageOps
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tempfile import NamedTemporaryFile

from tensorflow.keras.layers import Conv2D,Flatten,Dense,MaxPool2D,BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing.image import ImageDataGenerator,load_img
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import Model


st.title('Dynagraph Interpreter and Optimization Recommendation')

st.markdown('''

* This program is able to diagnose the SRP problem by reading the pump dynagraph 
* This problem is also able to give optimization recommendation to the SRP
* The program classify the dynagraph using deep machine learning algorithm (Tensorflow and Keras)


''')

st.write('**Dynagraph Interpreter**')

class_names = ['AirLock',
 'Broken Rod',
 'Gas Existence',
 'Inlet Valve Delay Closing',
 'Inlet Valve Leakage',
 'Outlet Sudden Unloading',
 'Outlet Valve Leakage',
 'Parrafin Wax',
 'Piston Sticking',
 'Plunger Fixed Valve Collision',
 'Plunger Guide Ring Collision',
 'Plunger Not Filled Fully',
 'Proper Work',
 'Rod Excesive Vibration',
 'Sand Problem',
 'Small Plunger',
 'Thick Oil',
 'Tubing Leakage',
 'Valve Leakage']

def import_and_predict(image_data, model):
    
    size = (180,180)    
    image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
    image = np.asarray(image)
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        #img_resize = (cv2.resize(img, dsize=(75, 75),    interpolation=cv2.INTER_CUBIC))/255.
        
    img_reshape = img[np.newaxis,...]
    
    prediction = model.predict(img_reshape)
        
    return prediction

model = keras.models.load_model('dynagraph_intepreter.h5')

st.sidebar.subheader('Dynagraph Image Input')

input = st.sidebar.selectbox('Choose Image: ',['Your Image','Sample Image'])

if input =='Your Image':
    uploaded_file = st.sidebar.file_uploader("Choose dynagraph file: ",type=['png','jpg'])

    if uploaded_file is None:
        image = Image.open('test/' + os.listdir('dataset')[0])
        st.image(image)
        
    else:
        st.image(uploaded_file)
        image = Image.open(uploaded_file)

else: 
    image_opt = st.sidebar.selectbox('Choose sample image: ',os.listdir('dataset'))
    
    image = Image.open('test/' + image_opt)
    st.image(image)


predictions = import_and_predict(image, model)
score = tf.nn.softmax(predictions[0])

st.success(
            "Based on the dynagraph, the pump most likely to have **{}** problem with a **{:.2f}** % confidence based on dataset."
            .format(class_names[np.argmax(score)], 100 * np.max(score)))





st.sidebar.subheader('Optimization Recommendation Input')

#Additional Parameter Input
OilGravity=st.sidebar.number_input("Oil Gravity (API) : ",value=30)
GasGravity=st.sidebar.number_input("Gas Gravity (API) : ",value=30)
PumpDepth=st.sidebar.number_input("Pump Depth (ft): ",value = 1999)
dpl=st.sidebar.number_input("Plunger Diameter (in.): ",value = 1)
dr=st.sidebar.number_input("Rod Diameter (in.): ",value=1)
RhoSteel=st.sidebar.number_input("Steel Density (lb/ft3): ", value = 122)
c=st.sidebar.number_input("C value (in.): ", value = 130)
h=st.sidebar.number_input("h value (in.): ", value = 10)
d1=st.sidebar.number_input("d1 (in.): ", value = 50)
d2=st.sidebar.number_input("d2 (in.): ", value = 10)
E=st.sidebar.number_input("Steel Modulus Young (psi): ", value = 20)
SurfaceStrokeLength=st.sidebar.number_input("Surface Stroke Length (in.): ", value = 30)
StrokeLength=st.sidebar.number_input("Theoritical Pump Stroke Length (in.): ", value = 40)
MaxStrokeLength=st.sidebar.number_input("Theoritical Pump maksimum Stroke Length (in.): ", value = 50)
DesignRate=st.sidebar.number_input("Design Rate (STB/ d): ", value = 60)
PumpSpeed=st.sidebar.number_input("Pump Speed (Strokes/ min): ", value = 100)

#SRP Unit Type
#print("Please choose your SRP Unit Type:")
#print("a. Conventional")
#print("b. Air-balanced")
Userinput=st.sidebar.selectbox("Please choose your SRP Unit Type: ",["Conventional","Air-balanced"])

if Userinput == 'Conventional':
    Userinput1 = 'a'
else:
    Userinput1 = 'b'

#PreCalculation
#Hydrocarbon Properties
SGoil=141.5/(OilGravity+131.5)
SGgas=141.5/(GasGravity+131.5)
Bo=1.3

#Area
Ar=(22/7)*dr*dr/4
Ap=(22/7)*dpl*dpl/4

#Pump Displacement
N= 0.1484*Ap*SurfaceStrokeLength*PumpSpeed
Kr=727.0008
PumpD= DesignRate*Bo//0.1484/Ap/SurfaceStrokeLength
#Fluid Load (Fo)
Fo=((0.8*SGoil*62.4)+(0.2*SGgas*0.08))*PumpDepth*Ap/144

#Weight Of Rods In Fluid (Wrf)
Wrf=RhoSteel*PumpDepth*Ar/144

#Total Load (Wrf + Fo)
TotalLoad=Wrf+Fo

#Volumetric Efficiency
Ev=DesignRate/N

#Actual Liquid Production Rate
ActualRate=0.1484*Ap*PumpD*StrokeLength*Ev/Bo

#Cyclic Load Factor
M=1+(c/h)

#Peak Polished Rod Load
#Conditional for Conventional and Air-balance
if Userinput1 == "a":
    F1=(StrokeLength*PumpD*PumpD*(1+(c/h)))/70471.2
elif Userinput1 == "b":
    F1=(StrokeLength*PumpD*PumpD*(1-(c/h)))/70471.2

PRLmax=Fo+((1.1+F1)*Wrf)

#Minimum Polished Rod Load
#Conditional for Conventional and Air-balance
if Userinput1 == "a":
    F2=(StrokeLength*PumpD*PumpD*(1-(c/h)))/70471.2
elif Userinput1 == "b":
    F2=(StrokeLength*PumpD*PumpD*(1+(c/h)))/70471.2

PRLmin=(0.65-F2)*Wrf

#Frictional Power
Pf=0.000000631*Wrf*SurfaceStrokeLength*(PumpD)

#Polished Rod Power
PRHP=SurfaceStrokeLength*PumpD*(PRLmax-PRLmin)/750000

#Name Plate Power
PNamePlate=Pf+PRHP

#Work Done by Pump
PumpWork=TotalLoad*Ev

#Work Done by Polished Rod
PRwork=((PRLmax+PRLmin)/2)+Fo+Wrf

#Pump Stroke Length
PumpStrokeLength=c*d2/d1

#Static Stretch
StaticStretch=Fo*PumpDepth/Ar/E

#Plunger Over Travel (EP)
EP=Fo*PumpDepth/Ap/E

#Maximum Torque
Torque=StrokeLength/4*((Wrf)+(2*StrokeLength*PumpD*PumpD*Wrf/70471.2))

#1/Kr
konstanta=1/Kr

#Fo/Skr
SKr=SurfaceStrokeLength/(konstanta)
X=Fo/(SKr)

#Counter Weight Required (CBE)
if Userinput1 == "a":
    CBE= (0.5*Fo)+Wrf*(0.9+(StrokeLength*PumpD*PumpD*c/(70471.2*h)))
elif Userinput1 == "b":
    CBE= (0.5*Fo)+Wrf*(0.9-(StrokeLength*PumpD*PumpD*c/(70471.2*h)))

#Counter Weight Position
CounterPosition=c

#Damping Factor
DampingFactor=(0.5+0.15)/2

#Stress (Max)
StressMax=PRLmax/Ar

#Stress (Min)
StressMin=PRLmin/Ar

#Recommendation Output
st.info('''

**Optimization Recommendation**

Frictional Power               :  {} Horse Power

Polished Rod Power             :  {} Horse Power

Name Plate Power               :  {} Horse Power

Work Done By Pump              :  {} lbf

Work Done By Polished Rod      :  {} lbf

Volumetric Efficiency          :  {}

Actual Liquid Production Rate  :  {} STB/day

Cyclic Load Factor             :  {}

Peak Polished Rod Load         :  {} lbf

Minimum Polished Rod Load      :  {} lbf

Pump Stroke Length             :  {} in

Static Stretch                 :  {} in

Plunger OverTravel (EP)        :  {} in

Fluid Load (Fo)                :  {} lbf

Weight Of Rods in Fluid (Wrf)  :  {} lbf

Total Load (Wrf + Fo)          :  {} lbf

Maximum Torque                 :  {} lbf.ins

Fo/SKr                         :  {}

1/Kr                           :  {}

CounterWeight Required (CBE)   :  {} lbf

Counter Weight Position        :  {} in

Damping Factor                 :  {}

Maximum Stress                 :  {} psi

Minimum Stress                 :  {} psi


'''.format(Pf,PRHP,PNamePlate,PumpWork,PRwork,Ev,ActualRate,M,PRLmax,PRLmin,PumpStrokeLength,StaticStretch,EP,Fo,Wrf,
            TotalLoad,Torque,X,konstanta,CBE,c, DampingFactor, StressMax, StressMin))


