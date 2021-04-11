import streamlit as st
# To make things easier later, we're also importing numpy and pandas for
# working with sample data.
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from plotly.offline import init_notebook_mode, iplot
import plotly.graph_objects as go
import streamlit.components.v1 as components
#from sklearn.ensemble import RandomForestClassifier
import pickle
import shap
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from PIL import Image
st.title("Covid19 Immunology app-a-thon")

components.html(
    """
<div style="font-family:Helvetica Neue; font-size:20px; line-height:1.7;color:slategray ">
<b>Objective</b>: To explore the relationship between personalized immune repertoires and COVID-19 
disease variables and associated factors. This exploration is part of <a href= "https://precision.fda.gov/challenges/12"> Precision FDA competition</a>.
    <br><br>
    Data transformation: For the analysis shown here, we have used 'Adaptive&ISB_Metadata' file and j_call, d_call, v_call data from annotated file.
    Data distribution and correlation through pandas-profiling.<br>
    <br>
</div>
<br>
""",
    height=250,
)
image1 = Image.open('data_process_1.png')
image2 = Image.open('data_process_2.png')
st.image(image1, use_column_width=True)
st.image(image2, use_column_width=True)
data = pd.read_parquet('data/v_call.parquet')
####Plot 1
components.html(
    """
<div style="font-family:Helvetica Neue; font-size:20px; line-height:1.7; color:slategray;">
<br>For count of intervention, Nasal Canula dominates in Baseline stage than Acute disease stage. Also other two classes Convalescent and Recovered have no contribution in following intervention.<br> Non-hispanic or Latino dominates over other ethinicity group.<br>
</div>
<br>""",
    height=250,
)
fig, axes = plt.subplots(1, 2, figsize=(10, 5), )
data_display = data[data.intervention != ' ']
sns.countplot(y="intervention", hue="disease_stage", data=data_display,  ax=axes[0]).set(title='Count of Intervention for each disease stage')
data_display = data[data['ethnicity']!=' ']
sns.countplot(x="ethnicity", hue="disease_stage", data=data_display,  ax=axes[1]).set(title='Count of ethnicity for each disease stage')
plt.xticks(rotation=30)
st.pyplot(fig)
components.html(
    """
<div style="font-family:Helvetica Neue; font-size:20px; line-height:1.7; color:slategray;">
<br>Out of all medical history category, Asthma, Hypertension and Diabetes patients' has higher count and contribution to Acute disease stage. <br>
</div>
<br>""",
    height=150,
)
fig, axes = plt.subplots(figsize=(5, 5) )
data_display = data[(data['medical_history']!=' ') & (data['medical_history'].isin(['Chronic hypertension','Asthma','Diabetes (Type 2), Chronic hypertension','Hypertension','Asthma, Chronic hypertension']))]
sns.countplot(y="medical_history", hue="disease_stage", data=data_display).set(title='top 5 medical illness for each disease stage')
st.pyplot(fig)

###Plot 2
components.html(
    """
<div style="font-family:Helvetica Neue; font-size:30px; line-height:1.7; color:slategray;">
<br><b>Analysis on v_call variables</b><br>
""",
    height=100,
)
features = [col for col in data if col.startswith('trbv')]
df = data.groupby('disease_stage')[features].sum().T.reset_index().rename(columns={'index': 'features'})
df_acute = df.sort_values('Acute', ascending=False).reset_index(drop=True).head(50)
df_Baseline = df.sort_values('Baseline', ascending=False).reset_index(drop=True).head(50)
df_Convalescent = df.sort_values('Convalescent', ascending=False).reset_index(drop=True).head(50)
df_Recovered = df.sort_values('Recovered', ascending=False).reset_index(drop=True).head(50)
components.html(
    """
<div style="font-family:Helvetica Neue; font-size:20px; line-height:1.7; color:slategray;">
<br>Observing below graph for highest 50 count of v_call variables for each class(acute, recovered, convalescent, baseline), Recovered and Acute class had higher counts compared to Baseline and Convalescent.<br>
<br>All the variables in Acute and Recovered class except 'trbv6-9' is present in Acute and not in Recovered class and 'trbv7-3*01' is present in Recovered class and not not in Acute class.<br>
</div>
<br>
""",
    height=300,
)
fig = make_subplots(
    rows=4, cols=1,
    subplot_titles=("count of v_call variables",))

fig.add_trace(go.Bar( y=df_acute['Acute'], x=df_acute["features"],  
                     marker=dict(color=df_acute['Acute'], coloraxis="coloraxis")),
              1, 1)
fig.add_trace(go.Bar( y=df_Baseline['Baseline'], x=df_Baseline["features"],  
                     marker=dict(color=df_Baseline['Baseline'], coloraxis="coloraxis")),
              2, 1)
fig.add_trace(go.Bar( y=df_Convalescent['Convalescent'], x=df_Convalescent["features"],  
                     marker=dict(color=df_Convalescent['Convalescent'], coloraxis="coloraxis")),
              3, 1)
fig.add_trace(go.Bar( y=df_Recovered['Recovered'], x=df_Recovered["features"],  
                     marker=dict(color=df_Recovered['Recovered'], coloraxis="coloraxis")),
              4, 1)
fig.update_layout(height=1200, width=1200, coloraxis=dict(colorscale='reds'), showlegend=False,plot_bgcolor='white')
fig.update_xaxes(ticks="outside", tickwidth=2,tickangle=45, tickcolor='crimson', ticklen=10,title_text="v_call categories")
fig.update_yaxes(title_text="Acute class count", row=1, col=1)
fig.update_yaxes(title_text="Baseline class count", row=2, col=1)
fig.update_yaxes(title_text="Convalescent class count", row=3, col=1)
fig.update_yaxes(title_text="df_Recovered class count", row=4, col=1)
#fig.show()
st.plotly_chart(fig)

df = data.groupby('intervention')[features].sum().T.reset_index().rename(columns={'index': 'features'}).drop(' ',axis=1)
df = df.set_index('features')
df_o2 = df[['Extracorporeal membrane oxygenation']].sort_values(['Extracorporeal membrane oxygenation'], ascending=False).head(10).style.background_gradient(cmap='OrRd')
df_ventilation = df[['Mechanical ventilation']].sort_values(['Mechanical ventilation'], ascending=False).head(10).style.background_gradient(cmap='OrRd')
df_ventilation_o2 = df[['Extracorporeal membrane oxygenation, Mechanical ventilation']].sort_values(['Extracorporeal membrane oxygenation, Mechanical ventilation'], ascending=False).head(10).style.background_gradient(cmap='OrRd')
df_nausal_canula = df[['Nasal cannula']].sort_values(['Nasal cannula'], ascending=False).head(10).style.background_gradient(cmap='OrRd')
components.html(
    """
<div style="font-family:Helvetica Neue; font-size:20px; line-height:1.7; color:slategray;">
<br>Below data shows that variables: 'trbv6-5', 'trbv6-6', 'trbv6-1' have higher count/occurences for all 4 interventions.<br>
</div>
<br>
""",
    height=100,
)
button = st.radio(
    "Select intervention",
 ('Extracorporeal membrane oxygenation', 'Mechanical ventilation', 'Extracorporeal membrane oxygenation, Mechanical ventilation','Nasal cannula'))
if button == 'Extracorporeal membrane oxygenation':
    st.write(df_o2)
elif button =='Mechanical ventilation':
    st.write(df_ventilation)
elif button=='Extracorporeal membrane oxygenation, Mechanical ventilation':    
    st.write(df_ventilation_o2)
elif button=='Nasal cannula':
    st.write(df_nausal_canula)

components.html(
    """
<div style="font-family:Helvetica Neue; font-size:20px; line-height:1.7; color:slategray;">
<br>Below is plot for total count of all v_call variables for different age groups.<br>
Age group 37-51 has highest occurence of values.
</div>
<br>
""",
    height=150,
)    
temp = data.groupby(['age_min','sex'])[features].sum().sum(axis=1).reset_index().rename(columns={0:'count'})
temp = temp[temp['sex']!=' ']
temp['bin']=np.where(temp['age_min']<=36,1,np.where((temp['age_min']>36) & (temp['age_min']<=51),2,np.where((temp['age_min']>51) & (temp['age_min']<66),3,4)))
temp['age'] = np.where(temp['bin']==1,'<36',np.where(temp['bin']==2, '37-51',np.where(temp['bin']==3,'52-66','>66')))
fig = px.bar(temp.groupby(['bin','sex','age'])['count'].sum().reset_index().sort_values('bin'), x="age", y="count", color="sex", title="count of v_call variables on age and sex")
st.plotly_chart(fig)
##SHAP plot for model
components.html(
    """
<div style="font-family:Helvetica Neue; font-size:20px; line-height:1.7; color:slategray;">
<br>Built model(Random forest) on v_call count variables where 25 features where used out of 191 using different methods like forward selection, recursive feature elimination and Information value.<br>
<br>Below is the graph for model interpretation which is known as SHAP value plot.<br>
SHAP value refers to the contribution of a feature value to a prediction.<br>
The larger the SHAP value magnitude, the more important the driver is.<br>
--'trbv6-7*01' : most important feature which contributes more to  Convalescent class.<br> 
--'trbv5-7*01: this feature contributes in order of -> Acute>Recovered>Convalescent>Baseline.'<br>
-- variables like 'trbv15-1' and 'trbv10-2' does not contribute to Baseline class.
</div>
<br>
""",
    height=450,
)
pkl_filename = "pickle_model_v_call.pkl"
with open(pkl_filename, 'rb') as file:
    rf = pickle.load(file)
features = {'trbv10-1*01','trbv10-2','trbv10-2*01','trbv11-2*01','trbv11-2*03','trbv11-3*03','trbv12-4*01',
 'trbv13-1*01','trbv18-1*01','trbv20-1*01','trbv20-1*04','trbv21-1*01','trbv23-1*01','trbv24-1*01','trbv25-1*01',
 'trbv26-1*01','trbv27-1*01','trbv29-1*03','trbv4-1*01','trbv4-2*01','trbv4-2*02','trbv5-1*01','trbv5-4*01','trbv5-7*01','trbv6-1*01','trbv6-2*01','trbv6-2*02',
 'trbv6-5*01','trbv6-6*01','trbv6-7*01','trbv7-1*01','trbv7-2*02','trbv7-5*01','trbv7-6*01','trbva-1*01','trbv15-1',
 'trbv5-1','trbv17-1','trbv26-1','trbv12-2','trbva/or9-2*01','trbv21/or9-2*01','trbv24/or9-2','trbv24/or9-2*03','trbv20/or9-2*03','trbv29/or9-2*01',
 'trbv25/or9-2*01','trbv8-1','trbv1-1','trbv22-1'}
class_names = data['disease_stage'].unique().tolist()
shap_values = shap.TreeExplainer(rf).shap_values(data[features])
fig, axes = plt.subplots()
shap.summary_plot(shap_values, data[features], class_names=class_names, max_display=10)
st.pyplot(fig)

###Plot 2 J CALL
data = pd.read_parquet('data/j_call.parquet')
data_display = data[data.sex != ' ']
features = ['trbj1-1*01','trbj1-2*01','trbj1-3*01','trbj1-4*01','trbj1-5*01','trbj1-6*01','trbj1-6*02','trbj2-1','trbj2-1*01','trbj2-2*01','trbj2-2p*01',
 'trbj2-3*01','trbj2-4*01','trbj2-5','trbj2-5*01','trbj2-6*01','trbj2-7*01','trbj2-7*02']
components.html(
    """
<div style="font-family:Helvetica Neue; font-size:30px; line-height:1.7; color:slategray;">
<br><b>Analysis on j_call</b><br>
""",
    height=100,
)

######
df = data.groupby('disease_stage')[features].sum().T.reset_index().rename(columns={'index': 'features'})
df_acute = df.sort_values('Acute', ascending=False).reset_index(drop=True).head(50)
df_Baseline = df.sort_values('Baseline', ascending=False).reset_index(drop=True).head(50)
df_Convalescent = df.sort_values('Convalescent', ascending=False).reset_index(drop=True).head(50)
df_Recovered = df.sort_values('Recovered', ascending=False).reset_index(drop=True).head(50)
components.html(
    """
<div style="font-family:Helvetica Neue; font-size:20px; line-height:1.7; color:slategray;">
<br>Observing below graph for all j_call variables for each class(acute, recovered, convalescent, baseline), Recovered and Acute class had higher counts compared to Baseline and Convalescent.<br>
</div>
<br>
""",
    height=200,
)
fig = make_subplots(
    rows=4, cols=1,
    subplot_titles=("count of j_call variables",))

fig.add_trace(go.Bar( y=df_acute['Acute'], x=df_acute["features"],  
                     marker=dict(color=df_acute['Acute'], coloraxis="coloraxis")),
              1, 1)
fig.add_trace(go.Bar( y=df_Baseline['Baseline'], x=df_Baseline["features"],  
                     marker=dict(color=df_Baseline['Baseline'], coloraxis="coloraxis")),
              2, 1)
fig.add_trace(go.Bar( y=df_Convalescent['Convalescent'], x=df_Convalescent["features"],  
                     marker=dict(color=df_Convalescent['Convalescent'], coloraxis="coloraxis")),
              3, 1)
fig.add_trace(go.Bar( y=df_Recovered['Recovered'], x=df_Recovered["features"],  
                     marker=dict(color=df_Recovered['Recovered'], coloraxis="coloraxis")),
              4, 1)
fig.update_layout(height=1200, width=1200, coloraxis=dict(colorscale='reds'), showlegend=False,plot_bgcolor='white')
fig.update_xaxes(ticks="outside", tickwidth=2,tickangle=45, tickcolor='crimson', ticklen=10,title_text="j_call categories")
fig.update_yaxes(title_text="Acute class count", row=1, col=1)
fig.update_yaxes(title_text="Baseline class count", row=2, col=1)
fig.update_yaxes(title_text="Convalescent class count", row=3, col=1)
fig.update_yaxes(title_text="Recovered class count", row=4, col=1)
#fig.show()
st.plotly_chart(fig)

df = data.groupby('intervention')[features].sum().T.reset_index().rename(columns={'index': 'features'}).drop(' ',axis=1)
df = df.set_index('features')
df_o2 = df[['Extracorporeal membrane oxygenation']].sort_values(['Extracorporeal membrane oxygenation'], ascending=False).head(10).style.background_gradient(cmap='OrRd')
df_ventilation = df[['Mechanical ventilation']].sort_values(['Mechanical ventilation'], ascending=False).head(10).style.background_gradient(cmap='OrRd')
df_ventilation_o2 = df[['Extracorporeal membrane oxygenation, Mechanical ventilation']].sort_values(['Extracorporeal membrane oxygenation, Mechanical ventilation'], ascending=False).head(10).style.background_gradient(cmap='OrRd')
df_nausal_canula = df[['Nasal cannula']].sort_values(['Nasal cannula'], ascending=False).head(10).style.background_gradient(cmap='OrRd')
components.html(
    """
<div style="font-family:Helvetica Neue; font-size:20px; line-height:1.7; color:slategray;">
<br> Below is data for intervention variables and count of j call where trbj2-7*01 and trbj2-7*01 has highest count in all intervention categories. <br>
</div>
<br>
""",
    height=200,
)
button1 = st.radio(
    "Select intervention for j_call",
 ('Extracorporeal_membrane_oxygenation', 'Mechanical_ventilation', 'Extracorporeal_membrane_oxygenation, Mechanical_ventilation','Nasal_cannula'))
if button1 == 'Extracorporeal_membrane_oxygenation':
    st.write(df_o2)
elif button1 =='Mechanical_ventilation':
    st.write(df_ventilation)
elif button1=='Extracorporeal_membrane_oxygenation, Mechanical_ventilation':    
    st.write(df_ventilation_o2)
elif button1=='Nasal_cannula':
    st.write(df_nausal_canula)

###Shap plot for J_call
components.html(
    """
<div style="font-family:Helvetica Neue; font-size:20px; line-height:1.7; color:slategray;">
<br>Shap plot for j_call<br>
‘trbj2-5*01 : most important feature which contributes more to  Convalescent class.<br>
‘trbj2-4*01’ & ‘trbj2-1*01’ does not contribute to baseline class.<br>
</div>
<br>
""",
    height=200,
)
rf = pickle.load(open('j_call_model.sav', 'rb'))

class_names = data['disease_stage'].unique().tolist()
shap_values = shap.TreeExplainer(rf).shap_values(data[features])
st.set_option('deprecation.showPyplotGlobalUse', False)
fig, axes = plt.subplots()
shap.summary_plot(shap_values, data[features], class_names=class_names, max_display=10)
st.pyplot(fig)
components.html(
    """
<div style="font-family:Helvetica Neue; font-size:20px; line-height:1.7; color:slategray;">
<br>Boxplot for 5 most important j_call variables obtained from model results with respect to gender<br>
We observe that Baseline values for Male gender is less compared to other classes<br>
</div>
<br>
""",
    height=200,
)

fig, axes = plt.subplots(1, 4, figsize=(15, 5), sharey=True)
sns.boxplot(x="sex", y='trbj2-3*01', hue="disease_stage",
                 data=data_display, linewidth=2.5, ax=axes[0])
sns.boxplot(  x="sex", y='trbj1-3*01', hue="disease_stage",
                 data=data_display, linewidth=2.5, ax=axes[1])
sns.boxplot(  x="sex", y='trbj1-5*01', hue="disease_stage",
                 data=data_display, linewidth=2.5, ax=axes[2])
sns.boxplot(  x="sex", y='trbj2-7*02', hue="disease_stage",
                 data=data_display, linewidth=2.5, ax=axes[3])
st.pyplot(fig)



####Plot D_Call
components.html(
    """
<div style="font-family:Helvetica Neue; font-size:30px; line-height:1.7; color:slategray;">
<br><b>Analysis on d_call</b><br>
""",
    height=100,
)
data = pd.read_parquet('data/d_call.parquet')
features = [col for col in data if col.startswith('trbd')]
######
df = data.groupby('disease_stage')[features].sum().T.reset_index().rename(columns={'index': 'features'})
df_acute = df.sort_values('Acute', ascending=False).reset_index(drop=True).head(50)
df_Baseline = df.sort_values('Baseline', ascending=False).reset_index(drop=True).head(50)
df_Convalescent = df.sort_values('Convalescent', ascending=False).reset_index(drop=True).head(50)
df_Recovered = df.sort_values('Recovered', ascending=False).reset_index(drop=True).head(50)
components.html(
    """
<div style="font-family:Helvetica Neue; font-size:20px; line-height:1.7; color:slategray;">
<br>Observing below graph for all d_call variables for each class(acute, recovered, convalescent, baseline), Recovered and Acute class had higher counts compared to Baseline and Convalescent.<br>
</div>
<br>
""",
    height=200,
)
fig = make_subplots(
    rows=4, cols=1,
    subplot_titles=("count of d_call variables",))

fig.add_trace(go.Bar( y=df_acute['Acute'], x=df_acute["features"],  
                     marker=dict(color=df_acute['Acute'], coloraxis="coloraxis")),
              1, 1)
fig.add_trace(go.Bar( y=df_Baseline['Baseline'], x=df_Baseline["features"],  
                     marker=dict(color=df_Baseline['Baseline'], coloraxis="coloraxis")),
              2, 1)
fig.add_trace(go.Bar( y=df_Convalescent['Convalescent'], x=df_Convalescent["features"],  
                     marker=dict(color=df_Convalescent['Convalescent'], coloraxis="coloraxis")),
              3, 1)
fig.add_trace(go.Bar( y=df_Recovered['Recovered'], x=df_Recovered["features"],  
                     marker=dict(color=df_Recovered['Recovered'], coloraxis="coloraxis")),
              4, 1)
fig.update_layout(height=1200, width=1200, coloraxis=dict(colorscale='reds'), showlegend=False,plot_bgcolor='white')
fig.update_xaxes(ticks="outside", tickwidth=2,tickangle=45, tickcolor='crimson', ticklen=10,title_text="d_call categories")
fig.update_yaxes(title_text="Acute class count", row=1, col=1)
fig.update_yaxes(title_text="Baseline class count", row=2, col=1)
fig.update_yaxes(title_text="Convalescent class count", row=3, col=1)
fig.update_yaxes(title_text="Recovered class count", row=4, col=1)
#fig.show()
st.plotly_chart(fig)

df = data.groupby('intervention')[features].sum().T.reset_index().rename(columns={'index': 'features'}).drop(' ',axis=1)
df = df.set_index('features')
df_o2 = df[['Extracorporeal membrane oxygenation']].sort_values(['Extracorporeal membrane oxygenation'], ascending=False).head(10).style.background_gradient(cmap='OrRd')
df_ventilation = df[['Mechanical ventilation']].sort_values(['Mechanical ventilation'], ascending=False).head(10).style.background_gradient(cmap='OrRd')
df_ventilation_o2 = df[['Extracorporeal membrane oxygenation, Mechanical ventilation']].sort_values(['Extracorporeal membrane oxygenation, Mechanical ventilation'], ascending=False).head(10).style.background_gradient(cmap='OrRd')
df_nausal_canula = df[['Nasal cannula']].sort_values(['Nasal cannula'], ascending=False).head(10).style.background_gradient(cmap='OrRd')
button1 = st.radio(
    "Select intervention for d_call",
 ('Extracorporeal membrane oxygenation_', 'Mechanical ventilation_', 'Extracorporeal membrane oxygenation_, Mechanical ventilation_','Nasal cannula_'))
if button1 == 'Extracorporeal membrane oxygenation_':
    st.write(df_o2)
elif button1 =='Mechanical_ventilation_':
    st.write(df_ventilation)
elif button1=='Extracorporeal_membrane_oxygenation, Mechanical_ventilation_':    
    st.write(df_ventilation_o2)
elif button1=='Nasal_cannula_':
    st.write(df_nausal_canula)

components.html(
    """
<div style="font-family:Helvetica Neue; font-size:20px; line-height:1.7; color:slategray;">
<br>Shap plot for d_call<br>
Top four features of d_call contribute more to ‘Acute’ class.<br>
‘trbd2-1*01’ is the highest contributing variable is also the significant contributing variable to Recovered stage of covid-19.<br>
</div>
<br>
""",
    height=200,
)
rf = pickle.load(open('d_call_model.sav', 'rb'))
features = ['trbd1-1', 'trbd1-1*01', 'trbd2-1', 'trbd2-1*01', 'trbd2-1*02']
class_names = data['disease_stage'].unique().tolist()
shap_values = shap.TreeExplainer(rf).shap_values(data[features])
st.set_option('deprecation.showPyplotGlobalUse', False)
fig, axes = plt.subplots(figsize=(5,5))
fig = shap.summary_plot(shap_values, data[features], class_names=class_names, max_display=10)
st.pyplot(fig)
