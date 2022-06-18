from sklearn.naive_bayes import BernoulliNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
import pandas as pd

path = 'SMSSpamCollection.txt'
names = ['labels', 'messages']
data = pd.read_csv(path, sep='\t', header=None, names=names)

data = data.replace({'ham': 0, 'spam': 1})  #
print('数据集展⽰：')
print(data)
print('\n----------------------------------\n')
X = data['messages']
y = data['labels']
x_train, x_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=123)
# 实现词袋模型
vector_nominal = CountVectorizer()
vector_burnout = CountVectorizer()


# 伯努利模型分类垃圾短信
train_matrix = vector_burnout.fit_transform(x_train)
test_matrix = vector_burnout.transform(x_test)
Bernoulli = BernoulliNB()
clm_bernoulli = Bernoulli.fit(train_matrix, y_train)
result_burnout = clm_bernoulli.predict(test_matrix)
print('伯努利模型的预测结果,类型，长度：')
print(result_burnout, type(result_burnout), result_burnout.shape)
print('伯努利模型的前⼀百个预测结果：')
print(result_burnout[0:100])
print('伯努利模型准确率评分：' + str(clm_bernoulli.score(test_matrix, y_test)))
