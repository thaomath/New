import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import sklearn
import lightgbm as lgb
from lightgbm import LGBMClassifier
import shap
from streamlit_shap import st_shap
import pickle
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

############################
# Configuration de la page #
############################
st.set_page_config(
        page_title='Dashboard Score du Client',
        layout="wide" )

# Définition de quelques styles css
st.markdown(""" 
            <style>
            body {font-family:'Roboto Condensed';}
            h1 {font-family:'Roboto Condensed';}
            h2 {font-family:'Roboto Condensed';}
            p {font-family:'Roboto Condensed'; color:Gray; font-size:1.125rem;}
            .css-18e3th9 {padding-top: 1rem; 
                          padding-right: 1rem; 
                          padding-bottom: 1rem; 
                          padding-left: 1rem;}
            .css-184tjsw p {font-family:'Roboto Condensed'; color:Gray; font-size:1rem;}
            </style> """, 
            unsafe_allow_html=True)

# Centrage de l'image du logo dans la sidebar
col1, col2, col3 = st.columns([1,1,1])
with col1:
    st.sidebar.write("")
with col2:
    image = Image.open('Data\\logo.png')
    st.sidebar.image(image, use_column_width="always")
with col3:
    st.sidebar.write("")

########################
# Lecture des fichiers #
########################
data_test_std = pd.read_csv('Data\\P7_data_test_20features_importance_std_sample.csv', sep=",")
data_test_interprete = pd.read_csv('Data\\P7_data_test_interprete_sample.csv', sep=",")

#################################################
#     Lecture de l'information d'un client      #
#################################################

liste_clients=list(data_test_interprete['SK_ID_CURR'].values)

seuil = 0.52

# Selection d'un client
ID_client = st.selectbox("Merci de saisir l'identifiant du client:", (liste_clients))

#st.selectbox("Pour information : Liste des identifiants possibles", liste_clients)
st.text("")
#ID_client = st.text_input("Veuillez entrer l'Identifiant d'un client")
#ID_client = int(ID_client)

#Récupération des informations du client
data_client = data_test_interprete[data_test_interprete.SK_ID_CURR==int(ID_client)]
col1, col2 = st.columns(2)
with col1:
    st.write('__Information de Client__')
    
    st.write('Genre:', data_client['Genre (H/F)'].values[0])
    st.write('Age :', data_client['Age'].values[0], "ans")
    st.write('Date de création de dossier : Il y a ', -data_client['Date de création de dossier'].values[0], "jours")
    st.write('Date enregistrement : Il y a ', -data_client['Date enregistrement'].values[0], "jours")
    st.write('Jours de travail : ', -data_client['Jours de travail'].values[0], "jours")
    st.write('Pourcentage de jours travaillés : ', data_client['Pourcentage de jours travaillés'].values[0],"%")

with col2:
    st.write('__Information de crédit__')
    
    st.write('Prix du bien :', data_client['Prix du bien'].values[0],"$")
    st.write('Annuités :', data_client['Annuités'].values[0],"$")
    st.write('Pourcentage de revevue :', data_client['Pourcentage de revevue'].values[0],"%")
    st.write('Taux de paiement :', data_client['Taux de paiement'].values[0])
    st.write('Ratio de remboursement :', data_client['Ratio de remboursement'].values[0])
    st.write('Ration de crédit de bien :', data_client['Ration de crédit de bien'].values[0])

    #lecture_description_variables()
# Titre 
st.markdown("""<h1 style="color:#772b58;font-size:2.3em;font-style:italic;font-weight:700;margin:0px;">
                Analyse univariée : </h1>
            """, unsafe_allow_html=True)
st.write("")
    
liste_variables = data_test_interprete.drop(['SK_ID_CURR'], axis=1).columns
liste_variables_without_sex = data_test_interprete.drop(['SK_ID_CURR', 'Genre (H/F)'], axis=1).columns

choix = st.selectbox("Merci de choisir une variable : ", (liste_variables))

for var in liste_variables : 
    if var == choix :
        if var == 'Genre (H/F)' : 
            fig, ax = plt.subplots(figsize=(8, 3))
            sns.histplot(data_test_interprete[var], color="orchid", bins=20)
            #ax.axvline(data_client[var].values[0], color="red", linestyle='dashed')
            #ax.set(title=var)#, xlabel='Revenu (USD)', ylabel='')
            st.pyplot(fig)
       
        if var != 'Genre (H/F)' :        
            col1, col2 = st.columns(2)
            with col1:
                fig, ax = plt.subplots(figsize=(8, 3))
                sns.histplot(data_test_interprete[choix], color="orchid", bins=20)
                ax.axvline(data_client[var].values[0], color="red", linestyle='dashed')
                #ax.set(title=var)#, xlabel='Revenu (USD)', ylabel='')
                st.pyplot(fig)
        
            with col2:
                fig, ax = plt.subplots(figsize=(8, 3))
                sns.boxplot(data=data_test_interprete, x=var, color="blue")
                ax.axvline(data_client[var].values[0], color="red", linestyle='dashed')
                #ax.set(title=var)#, xlabel='Revenu (USD)', ylabel='')
                st.pyplot(fig)  
                

#Analyse bivariée
# Titre 
st.markdown("""<h1 style="color:#772b58;font-size:2.3em;font-style:italic;font-weight:700;margin:0px;">
                Analyse bivariée : </h1>
            """, unsafe_allow_html=True)
st.write("")

col1, col2 = st.columns(2)
with col1:
    graph=['Quelle variable voulez-vous voir ?', 'Age', 'Taux de paiement', 'EXT_SOURCE_3', 'Ratio de remboursement',
           'EXT_SOURCES_MEAN', 'Annuités', 'Ration de crédit de bien', 'BUREAU_ACTIVE_DEBT_PERCENTAGE_MIN', 
           'EXT_SOURCE_1', 'APPROVED_CNT_PAYMENT_MEAN', 'EXT_SOURCES_MIN', 'INS_AMT_PAYMENT_SUM', 'Prix du bien', 
           'Date de création de dossier', 'EXT_SOURCE_2', 'Pourcentage de revevue', 'Date enregistrement',
           'Pourcentage de jours travaillés', 'Jours de travail']
    choix1 = st.selectbox("Analyses possibles", graph)
#*------------------------------------------------------------------------------------------*

with col2:
    sub_graph=['Par rapport à quelle variable ?', 'Age', 'Taux de paiement', 'EXT_SOURCE_3', 'Ratio de remboursement',
               'EXT_SOURCES_MEAN', 'Annuités', 'Ration de crédit de bien', 'BUREAU_ACTIVE_DEBT_PERCENTAGE_MIN', 
               'EXT_SOURCE_1', 'APPROVED_CNT_PAYMENT_MEAN', 'EXT_SOURCES_MIN', 'INS_AMT_PAYMENT_SUM', 'Genre (H/F)', 
               'Prix du bien', 'Date de création de dossier', 'EXT_SOURCE_2', 'Pourcentage de revevue', 
               'Date enregistrement', 'Pourcentage de jours travaillés', 'Jours de travail']
    sub_choix = st.selectbox("", sub_graph)
    
    for var1 in liste_variables_without_sex :
        if var1 == choix1 :
            st.write("data_client[var1] : ", data_client[var1].values[0])
            
            for var2 in liste_variables :
                if var2 == sub_choix :
                    st.write("data_client[var2] : ", data_client[var2].values[0])
                    st.write(" ")
                    
                    if var2 != 'Genre (H/F)' :            
                        fig, ax = plt.subplots(figsize=(10, 6))
                        plt.scatter(data_test_interprete[var1], data_test_interprete[var2])
                        ax.axvline(data_client[var1].values[0], color="red", linestyle='dashed')
                        ax.axhline(data_client[var2].values[0], color="red", linestyle='dashed')
                        plt.xlabel(var1)
                        plt.ylabel(var2)
                        st.pyplot(fig)
                
                    if var2 == 'Genre (H/F)' :
                        fig2, ax2 = plt.subplots(figsize=(10, 6))
                        sns.histplot(data=data_test_interprete, x=var1, hue="Genre (H/F)")
                        ax2.axvline(data_client[var1].values[0], color="red", linestyle='dashed')
                        st.pyplot(fig2)    
       

    #################################################
    # Lecture du modèle de prédiction et des scores #
    #################################################
# Loading model to compare the results
model_LGBM = pickle.load(open('Data\\model_complete.pkl','rb'))
    
# Score client    
X = data_test_std[data_test_std.SK_ID_CURR==int(ID_client)]
X = X.drop(['SK_ID_CURR'], axis=1)
probability_default_payment = model_LGBM.predict_proba(X)[:, 1]
score_value = round(probability_default_payment[0]*100, 2) 
if probability_default_payment >= seuil:
    prediction = "Prêt NON Accordé"
else:
    prediction = "Prêt Accordé" 

# Affichage du Score client 
# Titre 1
st.markdown("""<h1 style="color:#772b58;font-size:2.3em;font-style:italic;font-weight:700;margin:0px;">
                Score du client: </h1>
            """, unsafe_allow_html=True)
st.write("")

col1, col2 = st.columns(2)
with col2:
    original_title = '<p style="font-size: 20px;text-align: center;"> <u>Probabilité d\'être en défaut de paiement : </u> </p>'
    st.markdown(original_title, unsafe_allow_html=True)
    original_title = '<p style="font-family:Courier; color:BROWN; font-size:50px; text-align: center;">{}%</p>'.format((probability_default_payment[0]*100).round(2))
    st.markdown(original_title, unsafe_allow_html=True)

    original_title = '<p style="font-size: 20px;text-align: center;"> <u>Conclusion : </u> </p>'
    st.markdown(original_title, unsafe_allow_html=True)

    if prediction == "Prêt Accordé":
        original_title = '<p style="font-family:Courier; color:GREEN; font-size:70px; text-align: center;">{}</p>'.format(prediction)
        st.markdown(original_title, unsafe_allow_html=True)
    else :
        original_title = '<p style="font-family:Courier; color:red; font-size:70px; text-align: center;">{}</p>'.format(prediction)
        st.markdown(original_title, unsafe_allow_html=True)    
    
# Impression du graphique jauge
with col1:
    fig = go.Figure(go.Indicator(
                        domain = {'x': [0, 1], 'y': [0, 1]},
                        value = float(score_value),
                        mode = "gauge+number+delta",
                        title = {'text': "Score du client", 'font': {'size': 24}},
                        delta = {'reference': seuil*100, 'increasing': {'color': "#3b203e"}},
                        gauge = {'axis': {'range': [None, 100],
                                'tickwidth': 3,
                                'tickcolor': 'darkblue'},
                                'bar': {'color': 'white', 'thickness' : 0.3},
                                'bgcolor': 'white',
                                'borderwidth': 1,
                                'bordercolor': 'gray',
                                'steps': [{'range': [0, 20], 'color': '#e8af92'},
                                          {'range': [20, 40], 'color': '#db6e59'},
                                          {'range': [40, 60], 'color': '#b43058'},
                                          {'range': [60, 80], 'color': '#772b58'},
                                          {'range': [80, 100], 'color': '#3b203e'}],
                                'threshold': {'line': {'color': 'white', 'width': 8},
                                              'thickness': 0.8,
                                              'value': seuil*100 }}))

    fig.update_layout(paper_bgcolor='white',
                      height=400, width=500,
                      font={'color': '#772b58', 'family': 'Roboto Condensed'},
                      margin=dict(l=30, r=30, b=5, t=5))
    st.plotly_chart(fig, use_container_width=True)    
    
    
################################
# Explication de la prédiction #
################################
    # Titre 2
st.markdown("""
                <h1 style="color:#772b58;font-size:2.3em;font-style:italic;font-weight:700;margin:0px;">
                Explication du calcul:</h1>
                """, 
                unsafe_allow_html=True)
st.write("")
    

    # Calcul des valeurs Shap
explainer_shap = shap.TreeExplainer(model_LGBM)
shap_values = explainer_shap.shap_values(data_test_std.drop(labels="SK_ID_CURR", axis=1))


    # récupération de l'index correspondant à l'identifiant du client
    #idx = int(data_test[data_test_std['SK_ID_CURR']==ID_client].index[0])
idx = int(data_test_std.index[data_test_std['SK_ID_CURR']==ID_client].tolist()[0])
    
    # Graphique force_plot
st.write("Le graphique suivant permet de voir où se place la prédiction (f(x)) par rapport à la `base value`.") 
st.write("Nous observons également les variables qui augmentent la probabilité du client d'être \
            en défaut de paiement (en rouge) et celles qui la diminuent (en bleu), ainsi que l’amplitude de cet impact.")    


pd.set_option('display.max_colwidth', None)

    #st.dataframe(lecture_description_variables())

st_shap(shap.force_plot(explainer_shap.expected_value[1], 
                            shap_values[1][idx,:], 
                            data_test_std.drop(labels="SK_ID_CURR", axis=1).iloc[idx,:], 
                            link='logit',
                            figsize=(20, 8),
                            ordering_keys=True,
                            text_rotation=0,
                            contribution_threshold=0.05))
    # Graphique decision_plot
st.write("Le graphique ci-dessous est une autre manière de comprendre la prédiction. ")

st_shap(shap.decision_plot(explainer_shap.expected_value[1], 
                            shap_values[1][idx,:], 
                            data_test_std.drop(labels="SK_ID_CURR", axis=1).iloc[idx,:], 
                            feature_names=data_test_std.drop(labels="SK_ID_CURR", axis=1).columns.to_list(),
                            feature_order='importance',
                            feature_display_range=slice(None, -21, -1), 
                            link='logit'))
    

    
