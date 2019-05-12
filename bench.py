import pandas as pd
from squad_df import v1
from sklearn.feature_extraction.text import TfidfVectorizer
from word_hash import CharIdf


def build(df):
    df = df[["question", "context"]]
    passages = list(set(df.context.values))
    ptoi = {p: i for i, p in enumerate(passages)}
    dataset = []
    for _, row in df.iterrows():
        dataset.append((row.question, ptoi[row.context], row.context))
    return pd.DataFrame(dataset, columns=["question", "ctxid", "context"])


df = pd.DataFrame(list(v1))
df = df.loc[~df.is_train]
df = df.sample(df.shape[0])
df = df.reset_index()
df = df[:100]  # keep it small for now
df = build(df)

vec = CharIdf()
docs = list(set(df.context))
x = vec.fit_transform(docs)
qv = vec.transform(df.question.values)
result = pd.np.argmax(pd.np.einsum("kd,md->km", x, qv), axis=0)
print("charidf")
print(pd.np.mean(result == df.ctxid.values))

tfidf = TfidfVectorizer()
x = tfidf.fit_transform(docs).todense()
qv = tfidf.transform(df.question).todense()
tf_result = pd.np.argmax(pd.np.einsum("kd,md->km", x, qv), axis=0)
print("tfidf")
print(pd.np.mean(tf_result == df.ctxid.values))
