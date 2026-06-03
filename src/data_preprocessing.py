import logging
import pandas as pd 
import re 
from collections import Counter



class preprocessing:
    def __init__(self,data_path,oov,pad,words):
        self.data=pd.read_csv(data_path)
        self.data.drop(columns=["augmented","technique"], inplace=True)
        self.oov=oov
        self.pad=pad
        self.df=self.data.copy()
        self.words=words

    def clean_text(self,text):
        text=text.lower()
        text=re.sub(r'\d+', '', text)
        text=re.sub(r'[^\w\s]', '', text)
        return text
    
    def pre(self):
        self.df["clean_text"]=self.df["query"].apply(self.clean_text)
        return self.df
    
    def make_vocab(self,texts):
        count=Counter()
        for text in texts:
            count.update(text.split())
            vocab={self.oov:1,self.pad:0}
            for word, _ in count.most_common(self.words-2):
                vocab[word]=len(vocab)
        return vocab
    
    def save_data(self):
        self.df.to_csv(r"D:\__Projects\Toxic-content---classification\Data\augmented_processed_data\processed_data.csv")
    


    
