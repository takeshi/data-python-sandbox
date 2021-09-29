# %%
import numpy
from matplotlib import pyplot as ptl
year = numpy.arange(0,100,1)
data = numpy.arange(100,200,1) + numpy.random.rand(100)*50

ptl.scatter(year[0:49],data[0:49],c="red")
ptl.scatter(year[50:99],data[50:99],c="green")
ptl.show()

# %% Titinic Sandbox
import pandas

def init():
    import pandas
    train = pandas.read_csv("data/train.csv")
    test = pandas.read_csv("data/test.csv")
    return train,test

train,test = init()

# 前処理
train.Cabin = train.Cabin.map(lambda x : "Na" if(pandas.isna(x)) else x[0])
# train.Age = train.Age.map(lambda x: train.Age.mean() if(pandas.isna(x))else x)
train.Embarked = train.Embarked.map(lambda x: "Na" if(pandas.isna(x))else x)
#
# is_na = train.isna().sum()
# is_null = train.isna().sum()

train.Age = train.Age.map(lambda x: round(x,-1))

def count_percentage(columns):

    def percentage(x):
        return 100 * x / float(x.sum())

    count = train.groupby(columns).Survived.count()
    per = count.groupby(level=0).apply(percentage)
    return pandas.DataFrame({
    "count":count,
    "percentage":per
    })

cabin_data = count_percentage(["Cabin","Survived"])
pclass_data = count_percentage(["Pclass","Survived"])
age_data = count_percentage(["Age","Survived"])
sex_data = count_percentage(["Sex","Survived"])

print(train.columns)

# %% rolling sample
import pandas as pd
s = pd.Series(range(10))
for window in s.rolling(window=2):
    print(window)

# %% edpanding
# データ生成
import pandas as pd
import numpy as np
df = pd.DataFrame(np.random.randn(100, 5))
print(df.shape)  # (100, 5)

# 平均
mean = df.expanding().mean()

# 標準偏差
std = df.expanding().std()

# aggを使ってまとめて算出も可能。
mean_std = df.expanding().agg(["mean", "std"])