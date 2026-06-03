import pandas as pd 
import logging 
from sklearn.model_selection import train_test_split
import numpy as np


class feature:
    def __init__(self,processed_data_path,vocab,dict,length,oov,pad):
        self.df=pd.read_csv(processed_data_path)
        self.vocab=vocab
        self.length=50
        self.dict=10000
        self.oov="<oov>"
        self.pad="<pad>"
        

    def encode(self,text):
        tokens=text.split()[:self.length]
        encoded=[self.vocab.get(token,self.vocab[self.oov]) for token in tokens]
        if len(encoded)>self.length:
            return encoded[:self.length]
        else:
            return encoded+[self.vocab[self.pad]]*(self.length-len(encoded))
    
    def encode_text_and_coloums(self):
        self.df["encoded"]=self.df["clean_text"].apply(lambda x: self.encode(x))
        self.df["Toxic Category"]=self.df["Toxic Category"].map({"unsafe":1,"Safe":0,"Violent Crimes":2,"Non-Violent Crimes":3,"Elections":4,"Sex-Related Crimes":5,"Child Sexual Exploitation":6,"Suicide & Self-Harm":7})

        return self.df

    def spilt(self):
        x,y=np.array(self.df["encoded"].tolist()),np.array(self.df["Toxic Category"].tolist())
        x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42,stratify=y)
        return x_train,x_test,y_train,y_test

        
        
        
