# -*- coding: utf-8 -*-
"""
Created on Thu Apr 15 12:10:42 2021

@author: Martin
"""
import csv
from collections import Counter
import random
import time

#%% classe Data
class Data:
    def __init__(self,values,classe):
        self.values=values
        self.classe=classe
    
    def distEuclide(self,trainingData):
        somme=0
        for i in range(len(self.values)):
            somme=somme+(self.values[i]-trainingData.values[i])**2
        return (somme)**(1/2)

#%% load data
def ListData(filename):
    listData=[]
    with open(filename) as csv_file:
        csv_reader=csv.reader(csv_file,delimiter=',')
        for row in csv_reader:
            l=[]
            
            for i in range(len(row)-1):
                l.append(float(row[i]))
            newData=Data(l,row[len(row)-1])
            listData.append(newData)
    return listData

def FinalTestData(filename):
    listData=[]
    with open(filename) as csv_file:
        csv_reader=csv.reader(csv_file,delimiter=',')
        for row in csv_reader:
            l=[]
            
            for i in range(len(row)):
                l.append(float(row[i]))
            newData=Data(l,"unknown")
            listData.append(newData)
    return listData

                
#%% division de data en 2 listes        
def Division(listData):
    randomlist=random.sample(range(0,len(listData)),len(listData)//5)
    randomlist.sort(reverse=True)
    unknownData=[listData[i] for i in randomlist]
    for i in randomlist:
        listData.pop(i)
    return unknownData
    
#%% calc distance et sort by distance   
def CalcDist(unknownData,listData):
    listDist=[]
    for i in range (len(listData)):
        tpl=(unknownData.distEuclide(listData[i]),listData[i].classe)
        listDist.append(tpl)
    return sorted(listDist,key=lambda dist:dist[0])
    
#%% top k nearest neighbours + result
def Knn(k,listDist):
    knnList=[listDist[i] for i in range(k)]
    return knnList

def Result(knnList):
    classeList=[i[1] for i in knnList]
    c=Counter(classeList)
    return c.most_common(1)[0][0]

#%% Algo knn
def Algo(k):
    start=time.time()
    f = open("finalTest_mouly_martin.txt", "w")
    #chargement des données dans la liste d'entrainement
    datalist1=ListData("data.csv")
    datalist2=ListData("preTest.csv")
    datalist=datalist1+datalist2
    print("nb de données d'entraînement:",len(datalist))
    #chargement des valeurs des data à classifier
    unknowndata=FinalTestData("finalTest.csv")
    print("nb de données à classifier:",len(unknowndata))
    #scaling des data pour améliorer la précision
    Scaling(datalist,unknowndata)
    #pour chaque data de cette liste de data à classifier
    for data in unknowndata:
        #on calcule la distance euclidienne entre ce point et tous les autres points
        distance=CalcDist(data, datalist)
        #on garde les k plus proches voisins
        voisins=Knn(k,distance)
        #on voit quelle classe est la plus présente parmi ces voisins
        resultat=Result(voisins)
        f.write(str(resultat))
        f.write("\n")
    f.close()
    stop=time.time()
    print("temps d'execution:",stop-start,"s")
        
#%% Scaling
def Scaling(datalist,unknowndata):
    stats=[]
    for i in range(len(datalist[0].values)):
        l=[]
        for data in datalist:
            l.append(data.values[i])
        stats.append((min(l),max(l)))
    for i in range(len(datalist[0].values)):
        for data in datalist:
            data.values[i]=(data.values[i]-stats[i][0])/(stats[i][1]-stats[i][0])
    for i in range(len(unknowndata[0].values)):
        for data in unknowndata:
            data.values[i]=(data.values[i]-stats[i][0])/(stats[i][1]-stats[i][0])
            
#%% test
def TestAlgo(k):
    start=time.time()
    #chargement des données dans la liste d'entrainement
    datalist1=ListData("data.csv")
    datalist2=ListData("preTest.csv")
    datalist=datalist1+datalist2
    print("nb de données:",len(datalist))
    #chargement des valeurs des data à classifier
    unknowndata=Division(datalist)
    Scaling(datalist,unknowndata)
    while(k<8):
        counter=0 #pour calculer la précision
        #pour chaque data de cette liste de data à classifier
        for data in unknowndata:
            #on calcule la distance euclidienne entre ce point et tous les autres points
            distance=CalcDist(data, datalist)
            #on garde les k plus proches voisins
            voisins=Knn(k,distance)
            #on voit quelle classe est la plus présente parmi ces voisins
            resultat=Result(voisins)
            #résultats
            if resultat==data.classe:
                counter+=1
        stop=time.time()
        print(k,"voisins:",round(counter/len(unknowndata)*100,3),"% -- temps execution:",round(stop-start,3),"secondes")
        k+=1
        start=time.time()
