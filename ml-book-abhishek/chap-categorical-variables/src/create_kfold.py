import pandas as pd
from sklearn import model_selection

if __name__ == "__main__":
    
    # df = pd.read_csv("../../data/cat_train.csv", sep=",")
    df = pd.read_csv("../../data/adult.csv", sep=",")

    # create new column called kfold and fill it with -1
    df["kfold"] = -1

    # randomize the data
    df = df.sample(frac=1).reset_index(drop=True)
    print(df.head())
    # get labels
    # y = df.target.values
    y = df.income.values

    kf = model_selection.StratifiedKFold(n_splits=5)

    # fill the new kfold column
    for f, (t_, v_) in enumerate(kf.split(X=df, y=y)):
        # print("kfold ", type(f))
        # print("v ", f)
        df.loc[v_, "kfold"] = f

    # fold_dataset_name = "../../data/cat_train_folds.csv"
    fold_dataset_name = "../../data/adult_folds.csv"
    # save the new csv with kfold column
    df.to_csv(fold_dataset_name, index=False)


    # check the no. of examples in kfold

    df = pd.read_csv(fold_dataset_name, sep=",")
    print(df.kfold.value_counts())

    print(f"Check target distribution per fold \n")
    print(f"fold : 0")
    print(df[df.kfold == 0].income.value_counts())

    print(f"fold : 1")
    print(df[df.kfold == 1].income.value_counts())

    print(f"fold : 2")
    print(df[df.kfold == 2].income.value_counts())

    print(f"fold : 3")
    print(df[df.kfold == 3].income.value_counts())

    print(f"fold : 4")
    print(df[df.kfold == 4].income.value_counts())