import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
# st.set_option('deprecation.showPyplotGlobalUse', False)
# st.set_page_config(layout="wide")
st.title('MA Forecast')
start = '2019-01-01'
end = '2022-12-01'

@st.cache
def loadData():
    All_Shipments_1_df = pd.read_csv('All_Shipments_1_simple.csv')
    df = All_Shipments_1_df.copy() 

    j_df = pd.read_csv("InView___Updated_List_of_5_Mil_Generics_for_Fcst_Testing__1_18_23_.csv")[['Generic','Plnt','4 Month Trend']]
    j_df['Plant-Generic'] = j_df['Plnt'].astype(str)+'-'+j_df['Generic'].astype(str)
    j_df = j_df.drop(columns=['Plnt','Generic'])

    def Four_Month_MA(df,flag,tag):
        df1 = df.copy()
        df1 = df1.drop(columns=['Year','Month']).rename(columns={flag:tag})
        df1 = df1.groupby(['ds','Plant-Generic'])[tag].sum().unstack()
        mask = pd.DataFrame(index = pd.DatetimeIndex(pd.date_range(start=start, end=end, freq="MS"))) #####
        df1 = mask.join(df1).fillna(0) #
        df1.index.name = 'ds'
        df1 = df1.loc[:, (df1 != 0).any(axis=0)]

        def shift_f(df1, n):
            df_ = df1.shift(periods=n)
            if n==0: 
                df_ = df_.stack().reset_index().rename(columns={'level_1':'Plant-Generic',0:tag}).set_index(['ds','Plant-Generic'])
            else:
                df_ = df_.stack().reset_index().rename(columns={'level_1':'Plant-Generic',0:tag+'-'+str(n)}).set_index(['ds','Plant-Generic'])
            return df_

        dff = shift_f(df1, 0).join(shift_f(df1, 1)).join(shift_f(df1, 2)).join(shift_f(df1, 3)).join(shift_f(df1, 4))\
            .join(shift_f(df1, 5)).join(shift_f(df1, 6)).join(shift_f(df1, 7)).join(shift_f(df1, 8))  
        dff = dff.dropna().reset_index()
        return dff

    df = df[df['Plant'].isin([3803, 3809, 3811, 3833, 3835, 3841])]
    df = df.rename(columns={'Actual GI Date':'Date'})
    df['Date'] = pd.to_datetime(df['Date']);df = df[df['Date']>=pd.to_datetime(start)]
    df['Year'] = df['Date'].dt.year; df['Month'] = df['Date'].dt.month

    df = df.groupby(['Plant','Generic','Year','Month'])[['sum','count']].sum().reset_index()
    df['ds'] = pd.to_datetime(df['Year'].astype(str)+'-'+df['Month'].astype(str)+'-1')  
    df = df[df['ds']<=pd.to_datetime(end)]    
    df['Plant-Generic'] = df['Plant'].astype(str)+'-'+df['Generic'].astype(str)
    df = df.drop(columns=['Plant','Generic'])
    df = df[df['Plant-Generic'].isin(j_df['Plant-Generic'].unique())] ######################

    s = Four_Month_MA(df,'sum','y')
    c = Four_Month_MA(df,'count','c')
    sc = s.set_index(['ds','Plant-Generic']).join(c.set_index(['ds','Plant-Generic']))

    sc['score (y-1)'] = sc.apply(lambda x: 1 if x['y-1']>0 else 0, axis=1)
    sc['score (y-2)'] = sc.apply(lambda x: 1 if x['y-2']>0 else 0, axis=1)
    sc['score (y-3)'] = sc.apply(lambda x: 1 if x['y-3']>0 else 0, axis=1)
    sc['score (y-4)'] = sc.apply(lambda x: 1 if x['y-4']>0 else 0, axis=1)

    sc['score (c-1)'] = sc.apply(lambda x: 3 if x['y-1']>0 else 0, axis=1)
    sc['score (c-2)'] = sc.apply(lambda x: 4 if x['y-2']>0 else 0, axis=1)
    sc['score (c-3)'] = sc.apply(lambda x: 2 if x['y-3']>0 else 0, axis=1)
    sc['score (c-4)'] = sc.apply(lambda x: 1 if x['y-4']>0 else 0, axis=1)

    sc['score'] = sc.apply(lambda x: (x['score (y-1)']+x['score (y-2)']+x['score (y-3)']+x['score (y-4)'])\
                        *(x['score (c-1)']+x['score (c-2)']+x['score (c-3)']+x['score (c-4)']), axis=1)
    sc['4 Month Trend'] = sc.apply(lambda x: 'SPORADIC' if x['score']<10 else 'INTERMITTENT' if x['score']<14 \
                                else 'UP' if (x['y-1']<x['y-2']) & (x['y-2']< x['y-3']) \
                                else 'DOWN' if (x['y-1']>x['y-2']) & (x['y-2']> x['y-3']) \
                                else 'FLAT', axis=1)
    sc = sc.reset_index()
    sc = sc[sc['ds'].apply(lambda x:x.tz_localize(None))>=pd.to_datetime('2022-01-01')]
    sc['MA'] = (sc['y-1']+sc['y-2']+sc['y-3']+sc['y-4'])/4.0
    sc['EWM'] = sc[['y-8','y-7','y-6','y-5','y-4','y-3','y-2','y-1']].T.ewm(halflife=4).mean().iloc[-1]
    sc['SMAPE'] = sc.apply(lambda x: 0 if abs(x['y']+x['MA'])==0 \
                        else (x['y']-x['MA'])/abs(x['y']+x['MA']) if x['y']>=x['MA'] \
                        else (x['MA']-x['y'])/abs(x['y']+x['MA']), axis=1)
    sc['SMAPE_EWM'] = sc.apply(lambda x: 0 if abs(x['y']+x['EWM'])==0 \
                        else (x['y']-x['EWM'])/abs(x['y']+x['EWM']) if x['y']>=x['EWM'] \
                        else (x['EWM']-x['y'])/abs(x['y']+x['EWM']), axis=1)

    return sc

df = loadData().copy()
####################################
st.header('Model Performance Summary')
fig, ax = plt.subplots(figsize=(6, 3)); 
df[['ds', 'Plant-Generic','SMAPE']].set_index(['ds', 'Plant-Generic']).unstack().mean().hist()
st.pyplot(fig)
####################################
st.header('Model Segmentation')
dg = df[['ds', 'Plant-Generic','SMAPE']].set_index(['ds', 'Plant-Generic']).unstack().mean().reset_index()\
    .rename(columns={0:'SMAPE'})
dg['segmentation'] = dg['SMAPE'].apply(lambda x: 'SMAPE<=0.4' if x<=0.4 else 'SMAPE<=0.7' if x<=0.7 else 'SMAPE>0.7')
fig, ax = plt.subplots(figsize=(6, 6)); 
dg.groupby('segmentation')['Plant-Generic'].count().plot.pie(autopct='%1.1f%%')
st.pyplot(fig)
segs = dg['segmentation'].unique()
seg = st.selectbox('Select a segment', segs, index=2)
fig, ax = plt.subplots(figsize=(6, 2)); 
dh = df[df['Plant-Generic'].isin(dg[dg['segmentation']==seg]['Plant-Generic'])]
dj = dh.groupby(['Plant-Generic'])['y'].sum().sort_values(ascending=False)
ax = dj.iloc[:50].plot.bar(logy=True)
st.pyplot(fig)
####################################
st.header('Individual Plant-Generic Combination Analysis')
option = st.selectbox('Select a plant-generic combination', dj.index, index=0)
ss = st.slider('Safety Stock as multiples of MA', 0.0, 5.0, 0.2)
dh['MA+SS'] = dh['MA']*(1+ss)
dh['SMAPE_SS'] = dh.apply(lambda x: 0 if abs(x['y']+x['MA'])==0 \
                    else (x['y']-x['MA']*(1+ss))/abs(x['y']+x['MA']*(1+ss)) if x['y']>x['MA']*(1+ss) \
                    else 0 if x['y']<=x['MA']*(1+ss) and x['y']>=x['MA'] \
                    else (x['MA']-x['y'])/abs(x['y']+x['MA']), axis=1) 

fig, ax = plt.subplots(figsize=(6, 2)); 
ax = dh[dh['Plant-Generic']==option].set_index('ds')['MA'].plot( marker='o'); ax.legend() 
dh[dh['Plant-Generic']==option].set_index('ds')['MA+SS'].plot( marker='o'); ax.legend() 
dh[dh['Plant-Generic']==option].set_index('ds')['y'].plot( marker='o'); ax.legend() 
ax.set_title('Actual vs Moving Average Shipment'); ax.set_ylim(bottom=0)
st.pyplot(fig)
di = dh[dh['Plant-Generic']==option].set_index('ds')[['4 Month Trend']]
di.index = pd.Series(di.index).dt.month
st.table(di.T)
fig, ax = plt.subplots(figsize=(6, 2)); 
ax = dh[dh['Plant-Generic']==option].set_index('ds')['SMAPE'].plot( marker='o'); ax.legend() 
dh[dh['Plant-Generic']==option].set_index('ds')['SMAPE_SS'].plot( marker='o'); ax.legend() 
st.pyplot(fig)
