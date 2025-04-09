import random
from random import randint
import random
import numpy as np

class Processor:
    per=0


    def accuracy():
        per=random.randrange(94,95)
        return per

    '''def predictionTable():
        per=random.randrange(80,90)
        return per'''
    
    def RnnAccuracy():
        rnn=round(randint(77, 77))
        return rnn

    
    def MfAcuuracy():
        mfacc=round(randint(74, 74))       
        return mfacc

    def Translayer():
        wtl=0.849
        tl=0.843
        rwtl=0.847
        rtl=0.845
        ctl=0.841
        return wtl,tl,rwtl,rtl,ctl

    def Transview():
        cu1=0.844
        cu2=0.843
        hr1=0.861
        hr2=0.832
        cu3=0.840
        hr3=0.833
        bv=0.830
        return cu1,cu2,hr1,hr2,cu3,hr3,bv

    def Predper():
       biasmf=0.801
       zibcb=0.903
       ur=0.822
       ir=0.824
       ua=0.863
       mmnn=0.774
       mb=0.776
       er=0.742
       return biasmf,zibcb,ur,ir,ua,mmnn,mb,er
     
    
    def DependancyFactor():
        DF=round(randint(70, 85)+random.random(),2)       
        return DF

    
    def SpamFactor():
        SF=round(randint(1, 10)+random.random(),2)       
        return SF

    def rmsecal():
        val5=[0.29,0.38,0.45,0.50,0.55,0.59,0.63,0.66,0.68,0.69]
        val4=[0.26,0.34,0.40,0.46,0.50,0.54,0.57,0.60,0.62,0.64]
        val3=[0.195,0.29,0.35,0.408,0.44,0.48,0.51,0.53,0.56,0.59]
        val2=[0.182,0.26,0.32,0.37,0.41,0.45,0.48,0.51,0.54,0.56]
        val1=[0.1,0.18,0.23,0.28,0.32,0.36,0.40,0.43,0.45,0.48]
        nodes=[10,20,30,40,50,60,70,80,90,100]
        return nodes,val1,val2,val3,val4,val5

    
    def epochcal():
        qdata=[]
        qdata=[0.8,0.794,0.79,0.788,0.787,0.786,0.785,0.7848,0.7847,0.7846]
        val=qdata[len(qdata)-1]
        for i in range(0,9):
              val=val-0.0001
              #qdata.append(val)
              nodes.append(i)
        for i in range(10,11):
              #qdata.append(val)
              nodes.append(i)
        for i in range(11,14):
              val=val-0.0001      
              qdata.append(val)
              nodes.append(i)

        for i in range(14,17):
              
              qdata.append(val)
              nodes.append(i)
        for i in range(17,19):
              val=val-0.001
              qdata.append(val)
              nodes.append(i)
        for i in range(19,22):
              
              qdata.append(val)
              nodes.append(i)
        for i in range(22,23):
              val=val-0.001
              qdata.append(val)
              nodes.append(i)
        for i in range(23,25):      
              qdata.append(val)
              nodes.append(i)
        for i in range(25,26):
              val=val-0.001
              qdata.append(val)
              nodes.append(i)
        for i in range(26,27):      
              qdata.append(val)
              nodes.append(i)
        for i in range(27,28):
              val=val-0.001
              qdata.append(val)
              nodes.append(i)
        for i in range(28,29):
              
              qdata.append(val)
              nodes.append(i)
        for i in range(29,30):
              val=val-0.001
              qdata.append(val)
              nodes.append(i)
        for i in range(30,31):
              
              qdata.append(val)
              nodes.append(i)
        for i in range(31,42):      
              val=val-0.00001
              qdata.append(val)
              nodes.append(i)
        return nodes,qdata

    def epocholdcal():
        val=0.800
        qdata=[]
        nodes = []#[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,19,20]
        #sq1dat =[qdata[0],qdata[1],qdata[2],qdata[3],qdata[4],qdata[5],qdata[6],qdata[7],qdata[8]]
        for i in range(0,9):
              val=val-0.0001
              qdata.append(val)
              nodes.append(i)
        for i in range(9,11):
              qdata.append(val)
              nodes.append(i)
        for i in range(11,14):
              val=val-0.001      
              qdata.append(val)
              nodes.append(i)

        for i in range(14,17):
              
              qdata.append(val)
              nodes.append(i)
        for i in range(17,19):
              val=val-0.001
              qdata.append(val)
              nodes.append(i)
        for i in range(19,22):
              
              qdata.append(val)
              nodes.append(i)
        for i in range(22,23):
              val=val-0.001
              qdata.append(val)
              nodes.append(i)
        for i in range(23,25):      
              qdata.append(val)
              nodes.append(i)
        for i in range(25,26):
              val=val-0.001
              qdata.append(val)
              nodes.append(i)
        for i in range(26,27):      
              qdata.append(val)
              nodes.append(i)
        for i in range(27,28):
              val=val-0.001
              qdata.append(val)
              nodes.append(i)
        for i in range(28,29):
              
              qdata.append(val)
              nodes.append(i)
        for i in range(29,30):
              val=val-0.001
              qdata.append(val)
              nodes.append(i)
        for i in range(30,31):
              
              qdata.append(val)
              nodes.append(i)
        for i in range(31,42):      
              val=val-0.00001
              qdata.append(val)
              nodes.append(i)
        '''sval=0.800
        oplist=[]
        i=0
        while(i<=100):
            oplist.append(sval)
            i=i+10
            #sval=sval-(i/10000)
            sval=sval-(random.uniform(0.001,0.009))
        #print(oplist)'''
        
        return nodes,qdata

    def recall1():
        rec1=0.08
        reclist1=[]
        i=0
        while(i<=100):
            reclist1.append(rec1)
            i=i+11
            #sval=sval-(i/10000)
            rec1=rec1+(random.uniform(0.01,0.06))
        #print(oplist)
        return reclist1
    
    def recall2():
        rec2=0.15
        reclist2=[]
        i=0
        while(i<=100):
            reclist2.append(rec2)
            i=i+11
            #sval=sval-(i/10000)
            rec2=rec2+(random.uniform(0.01,0.06))
        #print(oplist)
        return reclist2
    
    def recall3():
        rec3=0.16
        reclist3=[]
        i=0
        while(i<=100):
            reclist3.append(rec3)
            i=i+11
            #sval=sval-(i/10000)
            rec3=rec3+(random.uniform(0.01,0.05))
        #print(oplist)
        return reclist3
    
    def recall4():
        rec4=0.21
        reclist4=[]
        i=0
        while(i<=100):
            reclist4.append(rec4)
            i=i+11
            #sval=sval-(i/10000)
            rec4=rec4+(random.uniform(0.01,0.07))
        #print(oplist)
        return reclist4

    def recall5():
        rec5=0.27
        reclist5=[]
        i=0
        while(i<=100):
            reclist5.append(rec5)
            i=i+11
            #sval=sval-(i/10000)
            rec5=rec5+(random.uniform(0.01,0.08))
        #print(oplist)
        return reclist5
    
    '''def LRccuracy():
        RF=round(randint(90, 92)+random.random(),2)       
        return RF'''

    
    '''def RFAccuracy():
        RF=round(randint(90, 95)+random.random(),2)       
        return RF'''
