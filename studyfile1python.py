# -*- coding: utf-8 -*-
"""
Created on Thu Jan 26 14:37:47 2023

@author: gobub
"""
""" Temp stufy file """
""" Zerodivisionerror"""

"""
a="kbjnf"
print(a)
type(True)

lst1=['a','b','c',2,4]                              #list
lst1.append('p')                                    #append
print(lst1)                          

lst2=['p','q','r']
lst1.extend(lst2)                                   #extend 
print(lst1)

print(len(lst1))

lst1[3]='NEw'                                       #indexing
print(lst1)

tpl1=(1,2.2,'ab')                                   #Tuples (is immutable)

fruits = ["apple", "banana", "cherry"]              #unpacking 
x, y, z = fruits

dict1={'name':'Gobu','age':21,'height':179.5} 
print(dict1.keys())                                 #Dictionaries
dict1['Sex']='Male'
print(dict1)
print(dict1['height'])

lst3=['a',2,2,4,3,'p'] 
lst1.extend(lst3)
print(lst1)
tempset=set(lst1)                                   #Set 
templist=list(tempset)                      
print(templist)                         

p=input("enter p ")
q=input("enter  q ")
r=input("enter r ")  

import random
print(random.randrange(1,10))                       #random     



if    (p<=q and q<=r):                              #if
       print("yep transitive")
elif  (p==q and q==r):
       print("All are same")
else:
       print("Not transitive")     

lst3=[]      
for i in range(4):                                  #for 
     print(" Enter number ",i+1)
     k=float(input())
     lst3.append(k)  
print(" Largrest number is ",max(lst3))    

dict4={"gobu":'m',"Sooraj":'m',"Keziah":'f',"Abik":'m'}
for name in dict4.keys() :
    print("Sex of ",name," is ",dict4[name]) 
    
print(-26//5)                                        #floor function
n=0 
while n<=12:                                         #while
    print("Pwer of 2 to ",n," is",2**n)
    n+=1      

def fn1(a,b):                                        #functions
    area=a*b
    return area
a=float(input("Enter length"))
b=float(input("Enter breadth"))  
print(" Area is ",fn1(a,b)) 

def fn2(a,b,c,pi=3.14):
    z=print(a,'x'x, '+' ,b,pi,'x',' +', c)
    return z
a=int(input(" Enter a "))
b=int(input(" Enter b "))     
c=int(input(" Enter c "))    
fn2(a, b, c)                      
 
import math
x=int(input("Enter x"))
v1=math.sin(x)
print(v1)                                      #import functions from module
from math import sin,cos
x=int(input("Enter x"))
v1=sin(x)
print(v1)        
from math import *                                 #import everything
x=int(input("Enter x"))
v1=sin(x)
print(v1)            

urllib: handles interfacing with URLs (e.g. scraping data from the
Web).
• datetime: various commands and definitions for working with
date and time data.    


class directors:                                   #class  
    def __init__(self,n,g,p):
        self.name=n
        self.genre=g
        self.preference=p
        
    def wutmake(self):
        if self.genre=="Comedy":
            print("Bro is funne")     
        elif self.genre=="Drama":
            print("bro is serious")
        else  :
            print("idk some other gnere")
            
    def pref(self):
        if(self.preference=='y'):
            print(" yea ill watch",self.name,"'s film")
        elif(self.preference=='n'):
            print("yeah imma skip",self.name,"'s film for now")
            
Shane=directors("Shane Balck","Comedy",'y')
Shane.wutmake()
Shane.pref()

PTA=directors("Paul Thomas Anderson","Drama",'n')
PTA.wutmake()
PTA.pref()                                                     

                        
                                    #numpy


                                         
"""                                         

import numpy as np
import numpy.ma as ma
import pandas as pd
import matplotlib.pyplot as plt
import os 


"""
ra1=np.array([1,2,3,4,5,6])
print(ra1[4])
ra2=ra1[ra1%2==0]
print(ra2[ra2>3])
ra3=np.ones(12)
ra4=ra3[-5:-1]
print(ra4)

print(np.isnan([np.nan,np.inf,np.pi,50]))       #isnan

ra1=np.arange(12)
ra2=ra1[3:10:2]                                 #slicing,steps
ra3=ra1[::3]
print(ra2,ra3) 

ra1=np.arange(2,13,2)
ra2=np.array([21.5,4.5,5.4])
ra3=np.concatenate([ra1,ra2],dtype=float)
ra4=np.append(ra1,[11,10,9,8,7])
print(ra3,ra4) 

ra1=np.zeros((5,5),dtype=int)    
ra1[2,2]=1
print(ra1,ra1.shape)
ra2=ra1.flatten()                               #flatten
print(ra2)
ra3=np.ones((10,2))
print(ra3.reshape((4,5))) 
ra1=np.ones(5)
ra2=np.zeros(5)
a=np.vstack((ra1,ra2))                          #vstack, hstack
b=np.hstack((ra1,ra2)) 
print(a,a.shape,'\n', b,b.shape) 
a=np.random.randn(3,3)
print(a)                                   
                                                 #where, greater,equal

a=np.random.uniform(low = 0, high = 100, size = 100)
b=np.where(np.greater(a, 15), 0, a)              
c=np.where(np.greater(a, 75) | np.less(a,25),0, a)
cleaned=np.where(np.isnan(a) | np.equal(a,0),-1,a)     # |, &
compressed=np.where(~np.isnan(a),a,0)             #compress(requires 1D arrays)
print(cleaned)    
a=np.array([1,2,3,4,5])
b=np.array([10,2,36,9,4])
print(a**2)
print(a+b)
print(np.dot(a,b))
print(np.sqrt(b))
print(np.mean(a))
print(np.std(b))
ra1=np.ones((2, 4))
print(np.sum(ra1,axis = None))
print(np.sum(ra1,axis=0))
print(np.sum(ra1,axis=1))
                                                   #masking
a = np.array([2, 2, 2, 2, 4, 6])                  
m = np.zeros(len(a))              
m[:3] = 1                                         #make first 3 values invalid                                                  
a_masked = ma.masked_array(a, mask=m)              
print(np.mean(a), " ", np.mean(a_masked))
b = np.array([2, 2, 14, 6, np.nan, 6])
b_masked = ma.masked_invalid(b)                    #mask nan or inf 
print(b_masked)       # or masked_a = ma.masked_where(np.isnan(a), a)
b_masked.set_fill_value(6)                         #Set fill value
print(b_masked.mean)
b_masked2=ma.masked_where(np.less(b,6) | np.greater(b,10), b)
print(b_masked2)   
a = np.random.uniform(low=0, high=100, size=(4, 4))
b = a.astype(int)                                #change data type of array
print(b) 
c=np.random.randint(low=0, high=100, size=(4, 4))
print(c)
d=np.random.normal(loc=0, scale=.5, size=3)
e=np.random.binomial(n=10, p=.3, size=3)
f=np.array([1, 2, 3, 4, 5, 6])
g=np.random.choice(f, size=4)      
print(c, g, d, e )     
i=np.random.choice(f, size=3, replace=False, p=None ) 

a = np.ones((4,4))                                  # read and write
np.save('testfile.npy', a)
b = np.load('testfile.npy')
print(b)    
#c = np.loadtxt('tesfile2.csv', delimiter=',', usecols=range(9))
c=np.genfromtxt('tesfile2.csv', delimiter=',')[:,:-1]
print(c)                                     
       

                               #matplotlib
                                                
x = np.linspace(-np.pi, np.pi, 100 )                        
plt.plot(x, np.sin(x), color='yellow', label='Sin(x)')              
plt.plot(x, np.cos(x), linestyle='dashed', color='red', label='cos(x)')
plt.xlabel('x')
plt.ylabel('sin(X)')
plt.title('test plot 1', size=20, weight='medium italic')
plt.legend(loc='upper left')
plt.savefig('testfig11.png')
plt.show()

c=os.getcwd()
print(c)                   """

"""
import numpy as np
import matplotlib.pyplot as plt
import os


x = np.linspace(-np.pi, np.pi, 100 )

plt.figure()
plt.plot(x, np.sin(x), label='sin(x)', color='red', linestyle='dashed')
plt.plot(x, np.cos(x), label='cos(x)', color='blue', linestyle='dotted')
plt.xlabel(' x ')
plt.ylabel(' trigs ')
plt.legend(loc='best')
plt.savefig('plsfuckingsave.png', dpi=150)   #savefig MUST be before plt.show
plt.show()
print(os.path.dirname(os.path.realpath(__file__)))

some_random_sin_points = np.sin(x) + np.random.normal(0, .2, size=100)
some_random_cos_points = np.cos(x) + np.random.normal(0, .2, size=100)
plt.plot(x, some_random_sin_points, marker='o', linestyle='None', 
         markerfacecolor='lightblue', markeredgecolor='black', markersize=4,
         label='noisy sin(x)') 
plt.plot(x, some_random_cos_points, marker='s', linestyle='None', 
         markerfacecolor='lightgreen', markeredgecolor='black', markersize=4,
         label='noisy cos(x)')
#linesty=None for scatter(not line), marker=o(circle),s(square)
plt.grid()
plt.legend(loc='best')
plt.savefig('plssave2.png')      

sample1 = np.random.normal(-1, 1, size=1000)
sample2 = np.random.normal(0, 1, size=1000)
plt.axis([-5, 5, 0, 150])              #manually set axes[xmin,xmax,ymin,ymax]
plt.hist(sample1,  bins=30, color='r', alpha=.69, label='hist1')
plt.hist(sample2,  bins=30, color='b', alpha=.69, label='hist2')
plt.xlabel('idk some x')
plt.ylabel('isk some y')
plt.grid()
plt.legend(loc='best')
plt.figure(figsize=(3,3))              #manually set figsize

sample1 = np.random.normal(-1, 1, size=1000)
sample2 = np.random.normal(1, .5, size=1000)
sample3 = np.random.normal(0, 1.5, size=1000)
sample4 = np.random.normal(-0.2, 2, size=1000)  

data = [sample1, sample2, sample3, sample4 ]

plt.subplots_adjust(hspace=.5, wspace=.5)  #increase spacing between plots

for i, d in enumerate(data):                           #enumerate
    plt.subplot(2, 2, i+1)
    plt.axis([-5, 5, 0, 75])
    plt.hist(d, bins=20, color='r', alpha=.69)
    plt.xlabel('sample'+ str((i+1)))
    plt.ylabel('number')

times2 = lambda x : 2*x                                #lambda funtion
print(times2(3))    
 
def circle():
    z = np.linspace(0, 2*np.pi, 1000)
    x=np.sin(z)
    y=np.cos(z)
    return x,y
plt.figure(figsize=(5,5))
x,y = circle()
plt.plot(x, y)
plt.title('Circle ?', weight='bold')
plt.show()

import pickle 
pickle.dump to save stuff into a file

maps                                                #look up examples
 
import datetime as dt
now = dt.datetime(2022, 2, 5, 23, 16, 00)
bday = dt.datetime(2001, 3, 14, 2, 30, 00 )
print(now-bday)                                     #datetime
"""


                                    #Pandas

"""
countries = [["Lithuania", 3.4e6, 38.3e9, 66.8],
["Australia", 20.6e6, 821e9, 88.6],
["United Kingdom", 60.0e6, 2772e9, 89.9],
["United States", 303.9e6, 13751e9, 81.4],
["China", 1331.4e6, 3206e9, 42.2]]

df1 = pd.DataFrame(data=countries,                
                   columns=("Countries", "Population", 
                            "GDP", "Urban popuation"))

print(df1, type(df1))
gdp = df1["GDP"]
df1["GDP per capita"] = df1["GDP"] / df1["Population"]
print(gdp, df1["GDP per capita"])  
print(df1)
print(df1.iloc[2], df1.iloc[2:3,2])                 #print rows, cells
voda=pd.read_csv("Vodaphone.csv", index_col=0)
print(voda)    
voda["Chumma oru column"] = np.nan
print(voda)
voda.to_csv("Vodaphone 2.csv")   
lst1=df1.values.tolist()                            #convert to other dtypes
print(lst1)           
df2=pd.read_excel("Barclays.xlsx")

plt.figure()
plt.plot(df2["Day count"], df2["Price"])
plt.xlabel("Number of Days")
plt.ylabel(" Prices ")
plt.show()   
ls1 = [1, 2, 3, 4, 5, 6, 7, 8, 9]
ls2 = ls1[-2: -4: -1]                   3 use negative steps for
print(ls2)    
"""


lst1 = ['a', 'b', 'c', 'd', 'e']        #loop enumerate
for index,item in enumerate(lst1):
    print(item, "is at position", index+1)
    
# for loop in Ndarray, use np.nditer(ra)  
# for loop in dataframe, use df1.iterrows()
# for a,b in df1.iterrows()













