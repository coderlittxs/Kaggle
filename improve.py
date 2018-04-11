import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
pd.set_option("display.width",1600)
pd.set_option("max_colwidth",400)
train=pd.read_csv("train.csv")
test=pd.read_csv("test.csv")
PassengerId=test["PassengerId"]
all_data=pd.concat([train,test],ignore_index=True)
# print(train.head)
# print(train.info())
# print(train["Survived"].value_counts())
# 1)Sex Feature：女性幸存率远高于男性
# sns.barplot(x="Sex",y="Survived",data=train)
# plt.show()
# 2)Pclass Feature：乘客社会等级越高，幸存率越高
# sns.barplot(x='Pclass',y='Survived',data=train)
# plt.show()
# 3)SibSp Feature：配偶及兄弟姐妹数适中的乘客幸存率更高
# sns.barplot(x='SibSp',y='Survived',data=train)
# plt.show()
# 4)Parch Feature：父母与子女数适中的乘客幸存率更高
# sns.barplot(x='Parch',y='Survived',data=train)
# plt.show()
# 5)从不同生还情况的密度图可以看出，在年龄15岁的左侧，生还率有明显差别，密度图非交叉区域面积非常大，但在其他年龄段，则差别不是很明显，认为是随机所致，因此可以考虑将此年龄偏小的区域分离出来。
# facet=sns.FacetGrid(train,hue="Survived",aspect=2)
# facet.map(sns.kdeplot,'Age',shade=True)
# facet.set(xlim=(0,train['Age'].max()))
# facet.add_legend()
# plt.xlabel('Age')
# plt.ylabel('density')
# plt.show()
# 6)Embarked登港港口与生存情况的分析 结果分析:C地的生存率更高,这个也应该保留为模型特征.
# sns.barplot(x='Embarked',y='Survived',data=train)
# plt.show()
# 7)Title Feature(New)：不同称呼的乘客幸存率不同
all_data['Title']=all_data['Name'].apply(lambda x:x.split(',')[1].split('.')[0].strip())
Title_dict={}
Title_dict.update(dict.fromkeys(['Capt','Col','Major','Dr','Rev'],'Officer'))#capt船长，col出口，major少校，officer军官
Title_dict.update(dict.fromkeys(['Don','Sir','the Countess','Dona','Lady','Master'],'Royalty'))#Royalty皇室，Don大学教师，Sir爵士，the Countess女伯爵，Dona，Lady贵妇，
Title_dict.update(dict.fromkeys(['Mme','Ms','Mrs'],'Mrs'))#Mme，Mrs太太
Title_dict.update(dict.fromkeys(['Mlle','Miss'],'Miss'))#Mlle小姐
Title_dict.update(dict.fromkeys(['Mr','Jonkheer'],'Mr'))#Mr先生
all_data['Title']=all_data['Title'].map(Title_dict)#使用map快速匹配字典的key
# sns.barplot(x='Title',y='Survived',data=all_data)
# plt.show()
# 8)FamilyLabel Feature(New)：家庭人数为2到4的乘客幸存率较高
# 新增FamilyLabel特征，先计算FamilySize=Parch+SibSp+1，然后把FamilySize分为三类。
all_data['FamilySize']=all_data['SibSp']+all_data['Parch']+1
# sns.barplot(x='FamilySize',y='Survived',data=all_data)
# plt.show()
def Fam_label(s):
    if(s>=2)&(s<=4):
        return 2
    elif (s==1)|((s>4)&(s<=7)):
        return 1
    elif (s>7):
        return 0
all_data['FamilyLabel']=all_data['FamilySize'].apply(Fam_label)
# sns.barplot(x='FamilyLabel',y='Survived',data=all_data)
# plt.show()
# 9)Deck Feature(New)：不同甲板的乘客幸存率不同
all_data['Cabin']=all_data['Cabin'].fillna('Unknown')
all_data['Deck']=all_data['Cabin'].str.get(0)
# sns.barplot(x='Deck',y='Survived',data=all_data)
# plt.show()
# 10)TicketGroup Feature(New)：与2至4人共票号的乘客幸存率较高
# 新增TicketGroup特征，统计每个乘客的共票号数。
Ticket_Count=dict(all_data["Ticket"].value_counts())
all_data['TicketGroup']=all_data['Ticket'].apply(lambda x:Ticket_Count[x])
# sns.barplot(x='TicketGroup',y='Survived',data=all_data)
# plt.show()
def Ticket_Label(s):
    if(s>=2)&(s<=4):
        return 2
    elif(s==1)|((s>4)&(s<=8)):
        return 1
    elif(s>8):
        return 0
all_data['TicketGroup']=all_data['TicketGroup'].apply(Ticket_Label)
# sns.barplot(x='TicketGroup',y='Survived',data=all_data)
# plt.show()
# 3.数据清洗
#
# 1)缺失值填充
#
# Age Feature：Age缺失量为263，缺失量较大，用三个特征构建随机森林模型，填充年龄缺失值。
from sklearn.ensemble import RandomForestRegressor
age_df=all_data[["Age","Title","Pclass","Sex"]]
age_df=pd.get_dummies(age_df)
known_age=age_df[age_df.Age.notnull()].as_matrix()
unknown_age=age_df[age_df.Age.isnull()].as_matrix()
y=known_age[:,0]
x=known_age[:,1:]
rfr=RandomForestRegressor(n_estimators=100,n_jobs=-1, random_state=None,)
rfr.fit(x,y)
pre_age=rfr.predict(unknown_age[:,1:])
all_data.loc[(all_data.Age.isnull()),"Age"]=pre_age
# 填充Embarked
# print(all_data[all_data["Embarked"].isnull()])
# print(all_data.groupby(by=['Pclass','Embarked']).Fare.median())
all_data['Embarked']=all_data['Embarked'].fillna('C')
# print(all_data[all_data["Embarked"].isnull()])
# Fare Feature：Fare缺失量为1，缺失Fare信息的乘客的Embarked为S，Pclass为3，所以用Embarked为S，Pclass为3的乘客的Fare中位数填充。
fare=all_data[(all_data['Embarked']=='S')&(all_data['Pclass']==3)].Fare.median()
all_data['Fare']=all_data['Fare'].fillna(fare)
# 2)同组识别
# 把姓氏相同的乘客划分为同一组，从人数大于一的组中分别提取出每组的妇女儿童和成年男性。
all_data['Surname']=all_data['Name'].apply(lambda x:x.split(',')[0].strip())
Surname_Count=dict(all_data['Surname'].value_counts())
all_data['FamilyGroup']=all_data['Surname'].apply(lambda x:Surname_Count[x])
Female_Child_Group=all_data.loc[(all_data['FamilyGroup']>=2)&((all_data['Age']<=12)|(all_data['Sex']=='Female'))]
Male_Ault_Group=all_data.loc[(all_data['FamilyGroup']>=2)&((all_data['Age']>12)&(all_data['Sex']=='male'))]
# 发现绝大部分女性和儿童组的平均存活率都为1或0，即同组的女性和儿童要么全部幸存，要么全部遇难
Female_Child=pd.DataFrame(Female_Child_Group.groupby('Surname')['Survived'].mean().value_counts())
Female_Child.columns=['GroupCount']
# print(Female_Child)
# 绝大部分成年男性组的平均存活率也为1或0
Female_Child_Group=Female_Child_Group.groupby('Surname')['Survived'].mean()
Dead_List=set(Female_Child_Group[Female_Child_Group.apply(lambda x:x==0)].index)
# print(Dead_List)
Male_Adult_List=Male_Ault_Group.groupby('Surname')['Survived'].mean()
Survived_List=set(Male_Adult_List[Male_Adult_List.apply(lambda x:x==1)].index)
# print(Survived_List)
# 为了使处于这两种反常组中的样本能够被正确分类，对测试集中处于反常组中的样本的Age，Title，Sex进行惩罚修改。
train=all_data.loc[all_data['Survived'].notnull()]
test=all_data.loc[all_data['Survived'].isnull()]
test.loc[(test['Surname'].apply(lambda x:x in Dead_List)),'Sex']='male'
test.loc[(test['Surname'].apply(lambda x:x in Dead_List)),'Age']=60
test.loc[(test['Surname'].apply(lambda x:x in Dead_List)),'Title']='Mr'
test.loc[(test['Surname'].apply(lambda x:x in Dead_List)),'Sex']='female'
test.loc[(test['Surname'].apply(lambda x:x in Dead_List)),'Age']=5
test.loc[(test['Surname'].apply(lambda x:x in Dead_List)),'Title']='Miss'
# 3)特征转换
# 选取特征，转换为数值变量，划分训练集和测试集。
all_data=all_data[['Survived','Pclass','Sex','Age','Fare','Embarked','Title','FamilyLabel','Deck','TicketGroup']]
all_data=pd.get_dummies(all_data)
train=all_data[all_data['Survived'].notnull()]
test=all_data[all_data['Survived'].isnull()].drop('Survived',axis=1)
x=train.as_matrix()[:,1:]
y=train.as_matrix()[:,0]
# 4.建模和优化
# from xgboost.sklearn import XGBClassifier
# clf=XGBClassifier(max_depth=6, learning_rate=0.1,
#                  n_estimators=28, silent=0,
#                  objective="binary:logistic", booster='gbtree',
#                  n_jobs=1, nthread=4, gamma=0.1, min_child_weight=1,
#                  max_delta_step=0, subsample=1, colsample_bytree=1, colsample_bylevel=1,
#                  reg_alpha=0, reg_lambda=1, scale_pos_weight=1,
#                  base_score=0.5, random_state=0, seed=None, missing=None)
# clf.fit(x,y)
# # from sklearn import svm
# # clf=svm.SVC()
# # clf.fit(x,y)
# # from sklearn.ensemble import BaggingClassifier
# from sklearn import cross_validation
# # from sklearn import linear_model
# # clf=linear_model.LogisticRegression()
# # bagging_clf=BaggingClassifier(clf)
# # bagging_clf.fit(x,y)
# # from sklearn.ensemble import GradientBoostingClassifier
# # gb=GradientBoostingClassifier()
# # gb.fit(x,y)
#
#  #简单看看打分情况
# a=cross_validation.cross_val_score(clf, x, y, cv=10)
# x=a.mean()
# print (a,x)
# #
# pre=clf.predict(test)
# 1)参数优化
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectKBest
# pipe=Pipeline([("select",SelectKBest(k=20)),("classify",RandomForestClassifier(random_state=10,max_depth=None, max_features='sqrt'))])
# param_test={'n_estimators':list(range(20,50,2)),'max_depth':list(range(3,60,3))}
# gsearch=GridSearchCV(estimator=pipe,param_grid=param_test,scoring="roc_auc",cv=10)
# param_test1 ={'n_estimators':range(20,100,2),'max_depth':range(10,100,2)}
# gsearch= GridSearchCV(estimator =RandomForestClassifier(max_features='sqrt',random_state=10),
#                        param_grid =param_test1,scoring='roc_auc',cv=10)
# gsearch.fit(x,y)
# print(gsearch.best_params_,gsearch.best_score_)
# 2)训练模型
from sklearn.pipeline import make_pipeline
# # 管道机制得以应用的根源在于，参数集在新数据集（比如测试集）上的重复使用。管道机制实现了对全部步骤的流式化封装和管理
select=SelectKBest(k=20)
clf=RandomForestClassifier(random_state=10,warm_start=True,n_estimators=19,max_depth=6)
pipeline=make_pipeline(select,clf)
print(pipeline.fit(x,y))
# 3)交叉验证
from sklearn import cross_validation,metrics
cv_score=cross_validation.cross_val_score(pipeline,x,y,cv=10)
print("CV Score:Mean-%.7g|Std-%.7g"%(np.mean(cv_score),np.std(cv_score)))
# 预测
# predictions=pipeline.predict(test)
#
# result = pd.DataFrame({'PassengerId':PassengerId, 'Survived':predictions.astype(np.int32)})
# result.to_csv(r"pre1.csv", index=False)