import s3fs
import pandas as pd
import numpy as np
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
import unicodedata
from PIL import Image
import math
from nltk.corpus import stopwords
import nltk
import unicodedata
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from nltk import word_tokenize


st.set_page_config(layout="wide")

fs = s3fs.S3FileSystem(anon=False, key=st.secrets["aws_access_key_id"], secret=st.secrets["aws_secret_access_key"])



nltk.download('stopwords')
nltk.download('punkt')

french_stopwords = set(stopwords.words('french'))
punctuations = '''!()-[]{};:'"\,<>./?@#$%^&*_~'''
filtre_stopfr =  lambda text: [''.join((c for c in unicodedata.normalize('NFD', token.lower()) if unicodedata.category(c) != 'Mn')) for token in text if (token.lower() not in french_stopwords) and (token.lower() not in punctuations)]




def remove_accents(input_str):
    nfkd_form = unicodedata.normalize('NFKD', input_str)
    return u"".join([c for c in nfkd_form if not unicodedata.combining(c)])

@st.cache
def summarize_dataset(df):
    toons = df.columns[1:]
    data = [[] for _ in range(5)]
    idx = ['Univers', 'Artistes', 'Genres', 'Support', 'Description']
    attributes = df['Unnamed: 0'].tolist()
    for t in toons:
        d = [[] for _ in range(5)]
        data_to_summarize = df[t].tolist()
        for i,a in enumerate(attributes):
            if a in idx and str(data_to_summarize[i])!= 'nan':
                if a =='Univers':
                    d[idx.index(a)] = remove_accents(data_to_summarize[i].lower())
                else:
                    d[idx.index(a)].append(remove_accents(data_to_summarize[i].lower()))
        for i in range(5):
            data[i].append(d[i])
    dico = {idx[i]: data[i] for i in range(5)}
    dico['Titre'] = toons
    return pd.DataFrame(dico)
   

def unique(df, column):
    liste = df[column].tolist()
    elements = []
    if column == 'Univers':
        return np.unique(df[column].to_numpy())
    else:
        for mini_liste in liste:
            for x in mini_liste:
                if x not in elements:
                    elements.append(x)
        return elements

def update_dict(dico, i):
    for k,v in dico.items():
        if len(v)<i:
            v.append(0)
        dico[k] = v

@st.cache
def compute_df(df, attributes):
    columns_name = []
    for a in attributes:
        unique_values = unique(df, a)
        for v in unique_values:
            columns_name.append(f'{a} - {v}')
    data = {c:[] for c in columns_name}
    for i in range(len(df)):
        for a in attributes:
            if a=='Univers':
                data[f"{a} - {df[a].iloc[i]}"].append(1)
            else:
                liste = df[a].iloc[i]
                for l in liste:
                    data[f'{a} - {l}'].append(1)
        update_dict(data, i+1)
    data['Titre'] = df['Titre'].tolist()
    return pd.DataFrame(data)

@st.cache
def create_df_similarity(df):
        X = df.drop(columns=['Titre']).to_numpy()
        similarity = cosine_similarity(X)
        df_res = pd.DataFrame(columns=df['Titre'].tolist(), data = similarity)
        df_res['Titre'] = df['Titre'].tolist()
        df_res.set_index('Titre', inplace=True)
        return df_res

@st.cache
def get_similar(toonTitre, df):
    if type(df)==float:
        return f'La somme des poids ne fait pas 1, elle fait {df}'
    else:
        df_res = df.loc[df.index == toonTitre].reset_index(). \
                melt(id_vars='Titre', var_name='Titre similaire', value_name='relevance'). \
                sort_values('relevance', axis=0, ascending=False)[1:6]
        return df_res


@st.cache
def create_df_similarity_weights(df_univers_support, df_genres, df_artistes, df_description, w_genre=0.4, w_description=0.4, w_support=0.1, w_artiste=0.1):
    if 0.999>w_genre + w_artiste + w_description + w_support or w_genre + w_artiste + w_description + w_support>1.01:
        return truncate(w_genre + w_artiste + w_description + w_support)
    else:
        X_univers_support = df_univers_support.drop(columns=['Titre']).to_numpy()
        similarity_univers_support = cosine_similarity(X_univers_support)

        X_genres = df_genres.drop(columns=['Titre']).to_numpy()
        similarity_genres = cosine_similarity(X_genres)

        X_artistes = df_artistes.drop(columns=['Titre']).to_numpy()
        similarity_artistes = cosine_similarity(X_artistes)

        X_description = df_description.drop(columns=['Titre']).to_numpy()
        similarity_description = cosine_similarity(X_description)

        similarity = similarity_univers_support * w_support + similarity_genres * w_genre + similarity_artistes * w_artiste + similarity_description * w_description
        df_res = pd.DataFrame(columns=df_univers_support['Titre'].tolist(), data = similarity)
        df_res['Titre'] = df_univers_support['Titre'].tolist()
        df_res.set_index('Titre', inplace=True)
        return df_res

def truncate(f):
    return math.floor(f * 10 ** 3) / 10 ** 3

@st.cache
def create_df_similarity_synopsis(df, column):
        count = CountVectorizer()
        count_matrix = count.fit_transform(df[column])
        similarity = cosine_similarity(count_matrix, count_matrix)
        df_res = pd.DataFrame(columns=df['Titre'].tolist(), data = similarity)
        df_res['Titre'] = df['Titre'].tolist()
        df_res.set_index('Titre', inplace=True)
        return df_res


@st.cache
def create_df_similarity_parcours(df, parcours):
    df2 = df[~df['Titre'].isin(parcours)]
    titres = df['Titre'].tolist()
    titres2 = df2['Titre'].tolist()
    df2 = df2.drop(columns=['Titre'])
    X = df.drop(columns=['Titre']).to_numpy()
    X2 = df2.to_numpy()
    Y = np.zeros(len(X[0]))
    for titre in parcours:
        Y += X[titres.index(titre)]
    Y = Y.reshape(1, -1)
    similarity = cosine_similarity(X2,Y)
    df_res = pd.DataFrame(similarity, columns = ['relevance'])
    df_res['Titre'] = titres2
    df_res = df_res.sort_values('relevance', axis=0, ascending=False)[1:6]
    return df_res

def calculate_similarity_parcours(df, parcours):
    df2 = df[~df['Titre'].isin(parcours)]
    titres = df['Titre'].tolist()
    df2 = df2.drop(columns=['Titre'])
    X = df.drop(columns=['Titre']).to_numpy()
    X2 = df2.to_numpy()
    Y = np.zeros(len(X[0]))
    for titre in parcours:
        Y += X[titres.index(titre)]
    Y = Y.reshape(1, -1)
    similarity = cosine_similarity(X2,Y)
    return similarity


def create_df_similarity_parcours(df_univers_support, df_genres, df_artistes, df_description, parcours, w_genre=0.4, w_description=0.4, w_support=0.1, w_artiste=0.1):
    if 0.999>w_genre + w_artiste + w_description + w_support or w_genre + w_artiste + w_description + w_support>1.01:
        return f"La somme des poids ne fait pas 1. Elle fait {truncate(w_genre + w_artiste + w_description + w_support)}"
    else:
        titres = df_univers_support[~df_univers_support['Titre'].isin(parcours)]['Titre'].tolist()
        similarity_univers_support = calculate_similarity_parcours(df_univers_support, parcours)
        similarity_artiste = calculate_similarity_parcours(df_artistes, parcours)
        similarity_description = calculate_similarity_parcours(df_description, parcours)
        similarity_genre = calculate_similarity_parcours(df_genres, parcours)
        similarity = similarity_univers_support * w_support + similarity_artiste * w_artiste + similarity_description * w_description + similarity_genre * w_genre
        df_res = pd.DataFrame(similarity, columns = ['relevance'])
        df_res['Titre'] = titres
        df_res = df_res.sort_values('relevance', axis=0, ascending=False)[1:6]
        return df_res


@st.cache(allow_output_mutation=True)
def read_file(filename):
    with fs.open(f"systeme-recommandation/Data/{filename}") as f:
        return pd.read_csv(f)

@st.cache
def read_file_synopsis(filename):
    with fs.open(f"systeme-recommandation/Data/{filename}") as f:
        return pd.read_csv(f, sep=';')

df = read_file("Thésaurus Krosmoz - Recommandations.csv")
df_synopsis_base = read_file_synopsis("descriptions_webtoons.csv")
df_synopsis = df_synopsis_base.copy()
df2 = summarize_dataset(df) 
df3 = compute_df(df2, ['Univers', 'Artistes', 'Genres', 'Support', 'Description'])
df_similarity = create_df_similarity(df3)

df_univers_support = compute_df(df2, ['Univers', 'Support'])
df_genres = compute_df(df2, ['Genres'])
df_artistes = compute_df(df2, ['Artistes'])
df_description = compute_df(df2, ['Description'])
df_similarity_weights = create_df_similarity_weights(df_univers_support, df_genres, df_artistes, df_description)


df_synopsis['mots_cles'] = df_synopsis['Desciption'].map(lambda x: ' '.join(filtre_stopfr( word_tokenize(x, language="french") )))
df_similarity_synopsis = create_df_similarity_synopsis(df_synopsis, 'mots_cles')

titres = df2['Titre'].tolist()

@st.cache(allow_output_mutation=True)
def load_img():
    images = []
    for titre in titres:
        with fs.open(f"systeme-recommandation/Data/IMG Webtoons/{titre}.jpg") as f:
            img = Image.open(f).convert('RGB')
            images.append(img)
    return images

images = load_img()


def get_images(reco_titres):
    reco_images = []
    for titre in reco_titres:
        if titre=='Chevalier Noir':
            titre='Chevalier noir'
        idx = titres.index(titre)
        reco_images.append(images[idx])
    return reco_images


option = st.sidebar.selectbox(
    'Quel WebToon ? ',
    titres
     )

df_recommand_synopsis = get_similar(option, df_similarity_synopsis)
titres_synopsis = df_recommand_synopsis['Titre similaire'].tolist()
relevances_synopsis = df_recommand_synopsis['relevance'].tolist()
images_synopsis = get_images(titres_synopsis)

df_without_weights = get_similar(option, df_similarity)
titres_without_weights = df_without_weights['Titre similaire'].tolist()
relevances_without_weights = df_without_weights['relevance'].tolist()
images_without_weights = get_images(titres_without_weights)


"WebToon choisi : "

col1, col2, col3, col4, col5 = st.columns(5)

with col3:
    st.image(images[titres.index(option)], caption = option)


with st.expander("Recommandation à partir du Synopsis"):
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.image(images_synopsis[0], caption = f"{titres_synopsis[0]}, Relevance : {truncate(relevances_synopsis[0])}")
    
    with col2:
        st.image(images_synopsis[1], caption = f"{titres_synopsis[1]}, Relevance : {truncate(relevances_synopsis[1])}")
    
    with col3:
        st.image(images_synopsis[2], caption = f"{titres_synopsis[2]}, Relevance : {truncate(relevances_synopsis[2])}")
    
    with col4:
        st.image(images_synopsis[3], caption = f"{titres_synopsis[3]}, Relevance : {truncate(relevances_synopsis[3])}")

    with col5:
        st.image(images_synopsis[4], caption = f"{titres_synopsis[4]}, Relevance : {truncate(relevances_synopsis[4])}")

    with st.expander("Documentation recommandation à partir du Synopsis"):
        text = """
        Ce système de recommandation se base sur les synopsis de chaque Webtoon. L’objectif pour un WebToon donné, est de trouver les œuvres ayant le synopsis le plus proche de celui de base.

Principe de fonctionnement : 
Pour pouvoir comparer les synopsis, il faut d’abord réussir à convertir le texte sous une forme mathématique qui permettrait de faire des comparaisons.
Tout d’abord, comme il peut y avoir des petites erreurs dans le texte (notamment des accents), on commence par supprimer tous les caractères spéciaux et toutes les majuscules.
Ensuite, il faut réussir à enlever tout le contenu du texte qui ne donne aucune information pertinente pour le système de recommandation, par exemple les connecteurs logiques.
Une liste de mots français inutiles est disponible et est utilisée pour supprimer le superflu. 

Exemple :
Le synopsis de Dofus Manga est : Arty est un jeune berger Féca. Il vit avec son grand-père Crail dans une région reculée d’Amakna. Séparé de son grand-père, Arty se voit confier une mission capitale : stopper le maléfice qui frappe les gelées de la forêt des Katrepat. Le terrible sortilège de « l’amour » les contraint à se multiplier sans cesse, au risque de recouvrir toute la surface de la province d’Amakna. C’est une lourde tâche pour un jeune garçon. Heureusement, il pourra compter sur l’aide de compagnons qu’il rencontrera en cours de route.
On obtient, après les différentes opérations de traitement, le texte suivant : 
arty jeune berger feca vit grand-pere crail region reculee ’ amakna separe grand-pere arty voit confier mission capitale stopper malefice frappe gelees foret katrepat terrible sortilege « ’ amour » contraint multiplier sans cesse risque recouvrir toute surface province ’ amakna ’ lourde tache jeune garcon heureusement pourra compter ’ aide compagnons ’ rencontrera cours route

Il ne reste plus qu’a mettre le résultat sous forme mathématique. Pour cela, on fait un référencement de tous les mots existant dans l’ensemble des synopsis. Cette liste de tous les mots correspond au vocabulaire disponible.
Il suffit ensuite de compter le nombre d’occurrence de chaque mot du vocabulaire dans le synopsis à convertir et de mettre ce résultat dans un vecteur.
Le vecteur final fait la taille du vocabulaire et pour un mot se trouvant en position i du vocabulaire, la valeur du vecteur en position i est le nombre d’occurrence du mot dans le texte.

Exemple : 

Prenons trois textes :
Texte_1 = « pere mange pomme »
Texte_2 = « mere mange banane »
Texte_3 = « pere achete banane »
Notre vocabulaire est : [pere, mere, mange, pomme, banane, achete]
Le texte 1 contient 1 fois les mots père, mange et pomme et ne contient pas les mots banane, mere, achete
On obtient le vecteur suivant vecteur_texte_1 = <1,0,1,1,0,0>
Pour les autres textes, on obtient :
Vecteur_texte_2 = <0,1,1,0,1,0>
Vecteur_texte_3 = <1,0,0,0,1,1>
Et voilà, nous avons convertit nos textes sous une forme mathématique !
Maintenant, il ne reste plus qu’à étudier la similarité entre les textes. Pour cela, nous allons utiliser la similarité cosinus. Concrètement, nous allons calculer la valeur de l’angle entre les vecteurs représentant deux textes.
La valeur que nous pouvons obtenir est situé entre 0 et 1, où 1 indique des vecteurs identiques et 0 des vecteurs opposés. Pour plus de détail : [Cosine similarity]( https://kanoki.org/2018/12/27/text-matching-cosine-similarity/)
Ainsi, pour un texte donné, il suffit de récupérer les autres textes qui ont une similarité la plus proche de 1 possible pour obtenir les textes les plus similaires.

Limites de cette méthode :
Cette méthode est un bon moyen de classifier des textes génériques (utilisation de termes communs) où le corpus contient des textes parlant de sujets très différents (exemple : Des textes qui parlent de foot et d’autres qui parlent de musique)

Dans notre cas, les synopsis contiennent beaucoup de mots spécifiques à l’univers d’Ankama, il n’y a pas de thématiques générales et il est très difficile de trouver de réelles similarités dans les textes.

Au-delà de ça, cette méthode reste limitée car ne considère pas comme identique les différentes variations d’un même mot. 
        """
        st.markdown(text)

        
with st.expander("Recommandation à partir du Thesaurus"):
    if st.checkbox('Sans poids'):
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.image(images_without_weights[0], caption = f"{titres_without_weights[0]}, Relevance : {truncate(relevances_without_weights[0])}")
        
        with col2:
            st.image(images_without_weights[1], caption = f"{titres_without_weights[1]}, Relevance : {truncate(relevances_without_weights[1])}")
        
        with col3:
            st.image(images_without_weights[2], caption = f"{titres_without_weights[2]}, Relevance : {truncate(relevances_without_weights[2])}")
        
        with col4:
            st.image(images_without_weights[3], caption = f"{titres_without_weights[3]}, Relevance : {truncate(relevances_without_weights[3])}")

        with col5:
            st.image(images_without_weights[4], caption = f"{titres_without_weights[4]}, Relevance : {truncate(relevances_without_weights[4])}")
    else:
        pass

    
        

    if st.checkbox('Avec poids'):
        with st.form("coucou"):
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                slider_genres = st.slider('Poids genre', 0.0, 1.0, 0.4,0.01,format="%.2f")
            with col2:
                slider_description = st.slider('Poids description', 0.0, 1.0, 0.4,0.01,format="%.2f")
            with col3:
                slider_support = st.slider('Poids univers+support', 0.0, 1.0, 0.1,0.01,format="%.2f")
            with col4:
                slider_artistes = st.slider('Poids artiste', 0.0, 1.0, 0.1,0.01,format="%.2f")
            submitted = st.form_submit_button("Nouvelle recommandation")
            if submitted:
                col1, col2, col3, col4, col5 = st.columns(5)
                df_with_weights = get_similar(option, create_df_similarity_weights(df_univers_support, df_genres, df_artistes, df_description,slider_genres, slider_description, slider_support, slider_artistes)
            )
                
                if type(df_with_weights)==str:
                    df_with_weights
                else:
                    titres_with_weights = df_with_weights['Titre similaire'].tolist()
                    relevances_with_weights = df_with_weights['relevance'].tolist()
                    images_with_weights = get_images(titres_with_weights)

                    with col1:
                        st.image(images_with_weights[0], caption = f"{titres_with_weights[0]}, Relevance : {truncate(relevances_with_weights[0])}")
                    
                    with col2:
                        st.image(images_with_weights[1], caption = f"{titres_with_weights[1]}, Relevance : {truncate(relevances_with_weights[1])}")
                    
                    with col3:
                        st.image(images_with_weights[2], caption = f"{titres_with_weights[2]}, Relevance : {truncate(relevances_with_weights[2])}")
                    
                    with col4:
                        st.image(images_with_weights[3], caption = f"{titres_with_weights[3]}, Relevance : {truncate(relevances_with_weights[3])}")

                    with col5:
                        st.image(images_with_weights[4], caption = f"{titres_with_weights[4]}, Relevance : {truncate(relevances_with_weights[4])}")
            else:
                pass
    else:
        pass


with st.expander("Recommandation à partir d'un profil utilisateur"):
    with st.form('ok'):
        
        parcours = st.multiselect(
        'Quels WebToons avez-vous aimés ?',
        titres)
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            slider_genres = st.slider('Poids genre', 0.0, 1.0, 0.4,0.01,format="%.2f")
        with col2:
            slider_description = st.slider('Poids description', 0.0, 1.0, 0.4,0.01,format="%.2f")
        with col3:
            slider_support = st.slider('Poids univers+support', 0.0, 1.0, 0.1,0.01,format="%.2f")
        with col4:
            slider_artistes = st.slider('Poids artiste', 0.0, 1.0, 0.1,0.01,format="%.2f")

        submit = st.form_submit_button("Nouvelle recommandation")
        if submit:
            col1, col2, col3, col4, col5 = st.columns([5 for _ in range(5)])
            df_parcours = create_df_similarity_parcours(df_univers_support, df_genres, df_artistes, df_description, parcours, slider_genres, slider_description, slider_support, slider_artistes)
            if type(df_parcours)==str:
                    df_parcours
            else:
                titres_parcours = df_parcours['Titre'].tolist()
                relevances_parcours = df_parcours['relevance'].tolist()
                images_parcours = get_images(titres_parcours)

                with col1:
                    st.image(images_parcours[0], caption = f"{titres_parcours[0]}, Relevance : {truncate(relevances_parcours[0])}")
                
                with col2:
                    st.image(images_parcours[1], caption = f"{titres_parcours[1]}, Relevance : {truncate(relevances_parcours[1])}")
                
                with col3:
                    st.image(images_parcours[2], caption = f"{titres_parcours[2]}, Relevance : {truncate(relevances_parcours[2])}")
                
                with col4:
                    st.image(images_parcours[3], caption = f"{titres_parcours[3]}, Relevance : {truncate(relevances_parcours[3])}")

                with col5:
                    st.image(images_parcours[4], caption = f"{titres_parcours[4]}, Relevance : {truncate(relevances_parcours[4])}")




