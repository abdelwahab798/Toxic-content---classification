from data_preprocessing import preprocessing

preprocess=preprocessing(r"D:\__Projects\Toxic-content---classification\Data\augmented_data\manual_few_shot_augmented.csv","<oov>","<pad>",10000)
df=preprocess.df
df=preprocess.pre()
vocab=preprocess.make_vocab(df["clean_text"])
preprocess.save_data()
print(len(vocab))
