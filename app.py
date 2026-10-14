import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
st.set_page_config(page_title='Churn Risk Demo',page_icon='🧠')
st.title('Customer Churn Risk — ML Demo')
st.warning('Educational demonstration using synthetic data only. Do not use it for real customer decisions.')
@st.cache_data
def make_data(n=1200):
 df=pd.DataFrame({'months_customer':[i%60+1 for i in range(n)],'monthly_fee':[25+((i*13)%900)/10 for i in range(n)],'support_tickets':[(i*7)%8 for i in range(n)],'plan':[['Basic','Standard','Premium'][i%3] for i in range(n)]})
 score=(df.support_tickets.ge(5).astype(int)*1.1+df.months_customer.lt(8).astype(int)*.8+df.monthly_fee.gt(85).astype(int)*.5+df.plan.eq('Basic').astype(int)*.4)
 df['churned']=((score+((pd.Series(range(n))%11)/20))>1.5).astype(int)
 return df
df=make_data(); X=df.drop(columns='churned'); y=df.churned
pre=ColumnTransformer([('num',StandardScaler(),['months_customer','monthly_fee','support_tickets']),('cat',OneHotEncoder(handle_unknown='ignore'),['plan'])])
model=Pipeline([('preprocess',pre),('classifier',LogisticRegression(max_iter=1000))])
Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.25,random_state=42,stratify=y); model.fit(Xtr,ytr); probs=model.predict_proba(Xte)[:,1]
a,b=st.columns(2); a.metric('Holdout accuracy',f'{accuracy_score(yte,model.predict(Xte)):.1%}')
try: b.metric('Holdout ROC-AUC',f'{roc_auc_score(yte,probs):.3f}')
except ValueError: b.info('ROC-AUC unavailable for this sample.')
st.subheader('Try a hypothetical customer')
months=st.slider('Months as customer',1,60,6); fee=st.slider('Monthly fee',20.,120.,75.,5.); tickets=st.slider('Support tickets',0,7,4); plan=st.selectbox('Plan',['Basic','Standard','Premium'])
one=pd.DataFrame([{'months_customer':months,'monthly_fee':fee,'support_tickets':tickets,'plan':plan}]); risk=float(model.predict_proba(one)[0,1]); st.metric('Demo churn score',f'{risk:.0%}'); st.progress(risk)
st.caption('Synthetic labels and metrics do not represent real-world performance. Validate on representative data before any practical use.')
with st.expander('View synthetic data'): st.dataframe(df.head(100),use_container_width=True)
