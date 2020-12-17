
"""
In Entity Embedding categories are represented as vectors

Entity embedding not only reduces memory usage
and speeds up neural networks compared with one-hot encoding, but more importantly by mapping
similar values close to each other in the embedding space it reveals the intrinsic properties of the
categorical variables (paper: Entity Embeddings of Categorical Variables)
    
We represent categories by vectors in both binarization
and one hot encoding approaches.

But what if we hav tens of thousands of categories.

This will create huge matrices and will take huge time
for us to train complicated models.

To eliminate this problem, we can represent vectors with 
float values instead

Idea is simple. you have an embedding layer for 
each categorical feature. So, every category in a column
can now be mapped to an embedding (like mapping words to embeddings in NLP)

You can then reshape these embeddings to their dimension to make
them flat and then concatenate all the flattened input embeddings.

Then at the end add a bunch of dense layers, an output layer. That's it
"""

import os
import joblib
import pandas as pd 
import numpy as np
from sklearn import metrics, preprocessing
from tensorflow.keras import layers, optimizers, callbacks, utils
from tensorflow.keras.models import Model, load_model
from tensorflow.keras import backend as K

def create_model(data, catcols):
    """
    This func. returns compiled keras model
    for entity embeddings
    :param data: this is pandas df
    :param catcols: list of categorical column names
    """

    # init list of input for embeddings
    inputs = []

    # init list of outputs for embeddings
    outputs = []

    # loop over all categorical columns
    for c in catcols:
        # find num of unique values in the column
        num_unique_values = int(data[c].nunique())
        # simple dimension of embeddings calculator
        # min size is half of the number of unique 
        
        # max size is 50. max size depends on the number of unique
        # categories too. 50 is quite sufficient most of the most
        # but if you have millions of unique values, you may 
        # need a larger dimension
        embed_dim = int(min(np.ceil((num_unique_values)/2), 50))

        # simple keras input layer with size 1
        inp = layers.Input(shape=(1,))

        # print(f"num unique values {num_unique_values}")
        # print(f"embed dim {embed_dim}")

        # add embedding layer to raw input
        # embedding size is always 1 more than unique values in input
        out = layers.Embedding(
            num_unique_values + 1, embed_dim, name=c
        )(inp)

        # print(f"output embeddings {out}")

        # 1-d spatial dropout is the standard for embedding laye
        out = layers.SpatialDropout1D(0.3)(out)


        # reshape nput to dimension of embedding
        # this becoms out output layer for current feature
        out = layers.Reshape(target_shape=(embed_dim, ))(out)

        # print(f"out of input shape {out.shape}")
        # this becomes our output layer for current feature

        # add input to input list
        inputs.append(inp)

        # add output to output list
        outputs.append(out)

    # concatenate all output layers
    # print(f"outputs {outputs}")

    x = layers.Concatenate()(outputs)

    # add a batchnorm layer
    # from here, everything is up to you
    # you can try different architectures
    # if you have numerical features, you should add
    # them here or in concatenate layer
    x = layers.BatchNormalization()(x)

    # here you network starts
    # a bunch of dense layers without dropout
    # start with 1 or two layers only
    x = layers.Dense(300, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    x = layers.BatchNormalization()(x)

    x = layers.Dense(500, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    x = layers.BatchNormalization()(x)

    x = layers.Dense(500, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    x = layers.BatchNormalization()(x)


    # using softmax and treating it as 2 class problem
    # you can also use sigmoid, then you need to use only one output class
    y = layers.Dense(2, activation="softmax")(x)

    # create find model
    model = Model(inputs=inputs, outputs=y)

    # compile the model
    # we use adam and binary cross entropy
    # feel free to use s.thng else and see how model behaves
    model.compile(loss="binary_crossentropy", optimizer="adam",
            metrics=['accuracy'])

    return model


def run(fold):
    
    df = pd.read_csv("../../data/cat_train_folds.csv", sep=",")

    features = [feature for feature in df.columns if feature not in ["id", "target", "kfold"]]

    # fill all nan with None
    # convert all feature to string type as all are categorical
    for feature in features:
        df.loc[:, feature] = df[feature].astype(str).fillna("NONE")

    for feature in features:
        le = preprocessing.LabelEncoder()
        # le.fit(df[feature])
        df.loc[:, feature] = le.fit_transform(df[feature].values)

    df_train = df[df.kfold.values != fold].reset_index(drop=True)
    df_valid = df[df.kfold.values == fold].reset_index(drop=True)

    # create tf.keras model
    model = create_model(df, features)
    # print(f"model summary {model.summary()}")
    # out features are lists of lists
    xtrain = [
        df_train[features].values[:, k] for k in range(len(features))
    ]
    xvalid = [
        df_valid[features].values[:, k] for k in range(len(features))
    ]

    # fetch target collumns
    ytrain = df_train.target.values
    yvalid = df_valid.target.values

    # convert target cols to categories
    # this is just binarization
    ytrain_cat = utils.to_categorical(ytrain)
    yvalid_cat = utils.to_categorical(yvalid)

    # fit the model
    model.fit(xtrain, ytrain_cat,
            validation_data=(xvalid, yvalid_cat),
            verbose=1,
            batch_size=1024,
            epochs=7)

    # generate validation predictions
    valid_preds = model.predict(xvalid)[:, 1]
    # print(f"valid preds {valid_preds[0][:, 1]}")

    # roc auc cruve
    # get roc auc score
    auc = metrics.roc_auc_score(yvalid, valid_preds)

    print(f"fold : {fold} auc : {auc}")

    # clear seession to free up some GPU memory
    K.clear_session() 


if __name__ == "__main__":
    
    total_folds = 5
    for fold in range(total_folds):
        run(fold)