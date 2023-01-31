import streamlit as st
import pandas as pd
All_Shipments_1_df = pd.read_csv('All_Shipments_1_simple.csv')
df = All_Shipments_1_df.copy() 

j_df = pd.read_csv("InView___Updated_List_of_5_Mil_Generics_for_Fcst_Testing__1_18_23_.csv")[['Generic','Plnt','4 Month Trend']]
j_df['Plant-Generic'] = j_df['Plnt'].astype(str)+'-'+j_df['Generic'].astype(str)
j_df = j_df.drop(columns=['Plnt','Generic'])

st.table(df)
st.table(j_df)
tab1, tab2, tab3 = st.tabs(["Cat", "Dog", "Owl"])

with tab1:
   st.header("A cat")
   st.image("https://static.streamlit.io/examples/cat.jpg", width=200)

with tab2:
   st.header("A dog")
   st.image("https://static.streamlit.io/examples/dog.jpg", width=200)

with tab3:
   st.header("An owl")
   st.image("https://static.streamlit.io/examples/owl.jpg", width=200)
