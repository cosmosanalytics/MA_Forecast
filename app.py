import streamlit as st
import pandas as pd
st.write('hello world')
All_Shipments_1_df = pd.read_csv('All_Shipments_1_simple.csv')
df = All_Shipments_1_df.copy() 

j_df = pd.read_csv("InView___Updated_List_of_5_Mil_Generics_for_Fcst_Testing__1_18_23_.csv")[['Generic','Plnt','4 Month Trend']]
j_df['Plant-Generic'] = j_df['Plnt'].astype(str)+'-'+j_df['Generic'].astype(str)
j_df = j_df.drop(columns=['Plnt','Generic'])
st.write(j_df)

# start = '2019-01-01'
# end = '2022-12-01'

# df = df[df['Plant'].isin([3803, 3809, 3811, 3833, 3835, 3841])]
# df = df.rename(columns={'Actual GI Date':'Date'})
# df['Date'] = pd.to_datetime(df['Date']);df = df[df['Date']>=pd.to_datetime(start)]
# df['Year'] = df['Date'].dt.year; df['Month'] = df['Date'].dt.month

# df = df.groupby(['Plant','Generic','Year','Month'])[['sum','count']].sum().reset_index()
# df['ds'] = pd.to_datetime(df['Year'].astype(str)+'-'+df['Month'].astype(str)+'-1')  
# df = df[df['ds']<=pd.to_datetime(end)]    
# df['Plant-Generic'] = df['Plant'].astype(str)+'-'+df['Generic'].astype(str)
# df = df.drop(columns=['Plant','Generic'])
# df = df[df['Plant-Generic'].isin(j_df['Plant-Generic'].unique())] ######################

# def Four_Month_MA(flag,tag):
#     df1 = df.copy()
#     df1 = df1.drop(columns=['Year','Month']).rename(columns={flag:tag})
#     df1 = df1.groupby(['ds','Plant-Generic'])[tag].sum().unstack()
#     mask = pd.DataFrame(index = pd.DatetimeIndex(pd.date_range(start=start, end=end, freq="MS"))) #####
#     df1 = mask.join(df1).fillna(0) #
#     df1.index.name = 'ds'
#     df1 = df1.loc[:, (df1 != 0).any(axis=0)]

#     df2 = df1.shift(periods=1); df3 = df2.shift(periods=1); df4 = df3.shift(periods=1); df5 = df4.shift(periods=1); 
#     df1 = df1.stack().reset_index().rename(columns={'level_1':'Plant-Generic',0:tag}).set_index(['ds','Plant-Generic'])
#     df2 = df2.stack().reset_index().rename(columns={'level_1':'Plant-Generic',0:tag+'-1'}).set_index(['ds','Plant-Generic'])
#     df3 = df3.stack().reset_index().rename(columns={'level_1':'Plant-Generic',0:tag+'-2'}).set_index(['ds','Plant-Generic'])
#     df4 = df4.stack().reset_index().rename(columns={'level_1':'Plant-Generic',0:tag+'-3'}).set_index(['ds','Plant-Generic'])
#     df5 = df5.stack().reset_index().rename(columns={'level_1':'Plant-Generic',0:tag+'-4'}).set_index(['ds','Plant-Generic'])

#     dff = df1.join(df2).join(df3).join(df4).join(df5) 
#     dff = dff.dropna().reset_index()
#     return dff

# s = Four_Month_MA('sum','y')
# c = Four_Month_MA('count','c')
# sc = s.set_index(['ds','Plant-Generic']).join(c.set_index(['ds','Plant-Generic']))

# sc['score (y-1)'] = sc.apply(lambda x: 1 if x['y-1']>0 else 0, axis=1)
# sc['score (y-2)'] = sc.apply(lambda x: 1 if x['y-2']>0 else 0, axis=1)
# sc['score (y-3)'] = sc.apply(lambda x: 1 if x['y-3']>0 else 0, axis=1)
# sc['score (y-4)'] = sc.apply(lambda x: 1 if x['y-4']>0 else 0, axis=1)

# sc['score (c-1)'] = sc.apply(lambda x: 3 if x['y-1']>0 else 0, axis=1)
# sc['score (c-2)'] = sc.apply(lambda x: 4 if x['y-2']>0 else 0, axis=1)
# sc['score (c-3)'] = sc.apply(lambda x: 2 if x['y-3']>0 else 0, axis=1)
# sc['score (c-4)'] = sc.apply(lambda x: 1 if x['y-4']>0 else 0, axis=1)

# sc['score'] = sc.apply(lambda x: (x['score (y-1)']+x['score (y-2)']+x['score (y-3)']+x['score (y-4)'])\
#                       *(x['score (c-1)']+x['score (c-2)']+x['score (c-3)']+x['score (c-4)']), axis=1)
# sc['4 Month Trend'] = sc.apply(lambda x: 'SPORADIC' if x['score']<10 else 'INTERMITTENT' if x['score']<14 \
#                               else 'UP' if (x['y-1']<x['y-2']) & (x['y-2']< x['y-3']) \
#                               else 'DOWN' if (x['y-1']>x['y-2']) & (x['y-2']> x['y-3']) \
#                               else 'FLAT', axis=1)

# df = sc.reset_index().copy()
# df = df[df['ds'].apply(lambda x:x.tz_localize(None))>=pd.to_datetime('2022-01-01')]
# df['MA'] = (df['y-1']+df['y-2']+df['y-3']+df['y-4'])/4.0

# df['SMAPE'] = df.apply(lambda x: 0 if abs(x['y']+x['MA'])==0 \
#                        else (x['y']-x['MA'])/abs(x['y']+x['MA']) if x['y']>=x['MA'] \
#                        else (x['MA']-x['y'])/abs(x['y']+x['MA']), axis=1)
# theta = 1.5
# df['SMAPE_SS'] = df.apply(lambda x: 0 if abs(x['y']+x['MA'])==0 \
#                        else (x['y']-x['MA']*theta)/abs(x['y']+x['MA']*theta) if x['y']>x['MA']*theta \
#                        else 0 if x['y']<=x['MA']*theta and x['y']>=x['MA'] \
#                        else (x['MA']-x['y'])/abs(x['y']+x['MA']), axis=1) 
####################
# df[['ds', 'Plant-Generic','SMAPE']].set_index(['ds', 'Plant-Generic']).unstack().mean().hist()

# dg = df[['ds', 'Plant-Generic','SMAPE_SS']].set_index(['ds', 'Plant-Generic']).unstack().mean().reset_index()\
#     .rename(columns={0:'SMAPE_SS'})
# dg['segmentation'] = dg['SMAPE_SS'].apply(lambda x: 'SMAPE_SS<=0.4' if x<=0.4 else 'SMAPE_SS<=0.7' if x<=0.7 else 'SMAPE_SS>0.7')
# dg.groupby('segmentation')['Plant-Generic'].count().plot.pie(autopct='%1.1f%%')

# df[df['Plant-Generic']=='3841-24816'].plot.line(x='ds', y='SMAPE', marker='o',figsize=(8,2))
# df[df['Plant-Generic']=='3841-24816'].plot.line(x='ds', y='y', marker='o',figsize=(8,2))

# df[df['Plant-Generic']=='3841-24816'][['ds','y','4 Month Trend','SMAPE']].T

# def MA(flag,ax): #this is by 'Plant-Generic'
#     ax = df[df['4 Month Trend']==flag].groupby(['Plant-Generic'])['SMAPE'].mean().hist(ax=ax,alpha=0.5, label=flag+' w/o SS')
#     ax.legend()
    
# def MA_SS(flag,ax): #this is by 'Plant-Generic'
#     ax = df[df['4 Month Trend']==flag].groupby(['Plant-Generic'])['SMAPE_SS'].mean().hist(ax=ax,alpha=0.5, label=flag+' w/ SS = 1*MA')
#     ax.legend()    
    
# fig, axes = plt.subplots(2,3,figsize=(20,10))
# flag = 'FLAT'; MA(flag,axes[0,0])
# flag = 'UP'; MA(flag,axes[0,1])
# flag = 'DOWN'; MA(flag,axes[0,2])
# flag = 'INTERMITTENT'; MA(flag,axes[1,0])
# flag = 'SPORADIC'; MA(flag,axes[1,1])

# flag = 'FLAT'; MA_SS(flag,axes[0,0])
# flag = 'UP'; MA_SS(flag,axes[0,1])
# flag = 'DOWN'; MA_SS(flag,axes[0,2])
# flag = 'INTERMITTENT'; MA_SS(flag,axes[1,0])
# flag = 'SPORADIC'; MA_SS(flag,axes[1,1])

# df.groupby(['Plant-Generic'])['4 Month Trend'].unique().reset_index().astype(str).groupby(['4 Month Trend'])['Plant-Generic'].unique()


