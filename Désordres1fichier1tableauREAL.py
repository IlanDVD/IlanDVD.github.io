#!/usr/bin/env python
# coding: utf-8

# # Recherche de désordres dans des documents techniques

# ## Importation de packages

# In[1]:


import sys
#sys.path.remove('C:\\Users\\idavid\\AppData\\Roaming\\Python\\Python38\\site-packages')
import os
import pandas as pd
import PyPDF2
import textract
import pdfplumber
import string
import nltk
from nltk.tokenize import *
from nltk.corpus import stopwords
from nltk.stem import *
from sklearn.feature_extraction.text import *
from os import walk
import re
import unidecode
import spacy
import fr_core_news_sm
from tqdm import tqdm_notebook as tqdm
import gensim
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
import gensim.downloader as api
import re
from difflib import SequenceMatcher
from io import BytesIO
import requests

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

'''
def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    base_path = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath("__file__")))
    return os.path.join(base_path, relative_path)
'''

#pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_colwidth', None, 'display.max_rows', None)
#pd.reset_option("display.max_rows", "display.max_columns")


# In[2]:


#get_ipython().system('python -m spacy download fr_core_news_sm')


# ## Récupération et traitement du texte
# ### Lecture du document pdf

# In[9]:


## Extraction du texte du pdf
def extractionPDF(fichier):
    i = 0
    final_extsplit = []
    df = pd.DataFrame()
    df['Text'] = 0
    rq = requests.get(fichier)

    with pdfplumber.open(BytesIO(rq.content)) as pdf:
        while i < len(pdf.pages):
            page = pdf.pages[i]
            ext = page.extract_text()
            if ext != None:
                for word in ext:
                    word = str(word)
                extsplit = ext.split('\n')
                for elem in extsplit:
                    extsplit2 = elem.split('.')
                    final_extsplit = final_extsplit + extsplit2


            i += 1
        df['Text'] = final_extsplit
    
    #print(df['Text'])
    df['Length'] = df.Text.str.len()
    df = df[df.Length > 1]
    df.drop('Length', axis=1, inplace=True)
    df = df.reset_index()
    df.drop('index', axis=1, inplace=True)
    
    #display(df)
    
    return df

## Reconstitution des phrases
def reconstruction(df):
    d = pd.DataFrame()
    d['Text'] = df['Text']
    s = ['l"','H ','n ']

    idx = 0
    for row in df['Text']:
        if row[0] == row[0].lower() and idx != 0 and row[0] not in string.punctuation and row[0:2] not in s :
            d['Text'][idx] = d['Text'][idx-1] + ' ' + d['Text'][idx] 
            d['Text'][idx-1] = None
        idx += 1

    d = d.dropna()
    d = d.reset_index()
    d.drop('index', axis=1, inplace=True)
    
    return d

## Reconstitution des phrases avancée par entité
def reconstructioEntités(d):
    nlp = spacy.load('fr_core_news_sm')
    #nlp = fr_core_news_sm.load
    _, words_list_type = listeMotsImportants('Type')
    data = pd.DataFrame()
    
    i = 0
    for elem in d['Text']:
        d['Text'][i] = unidecode.unidecode(elem)
        
        i += 1
        
    data['Text'] = d['Text']    
    
    idx = 0
    for r in d['Text']:
        doc = nlp(r)

        bigl = []
        i = 0

        tk = [token.text for token in doc]
        tag = [token.pos_ for token in doc]

        for t in tk:
            bigl.append([t, tag[i]])
            i += 1

        tab = pd.DataFrame(bigl, columns = ['Token','Pos-Tag'])
        
        #display(tab)

        l =  []    
        for ent in doc.ents:
            ll = (re.split('\W+', ent.text))
            for y in ll:
                l.append(y)
        
        for elem in l:
            if multi_re_find(words_list_type,elem) != None:
                l.remove(elem)
        
        if tab['Token'][0] in l and len(tab['Token'][0]) > 1:
            if idx != 0:
                data['Text'][idx] = data['Text'][idx-1] + ' ' + data['Text'][idx] 
                data['Text'][idx-1] = None


        idx += 1

    data = data.dropna()
    data = data.reset_index()
    data.drop('index', axis=1, inplace=True)
    
    #display(data)
    
    return data

### Recherche de désordres ###
## Fonction de cherche
def multi_re_find(patterns, phrase):
    i = 0
    
    for pattern in patterns:
        if len(re.findall(pattern,phrase)) > 0:
            return phrase
        
def multi_re_findPatt(patterns, phrase):
    i = 0
    
    for pattern in patterns:
        if len(re.findall(pattern,phrase)) > 0:
            return pattern

## Importation des 200 mots les plus récurrents au sein des descriptions de sinistres
def listeMotsImportants(col):
    tab = pd.read_excel("NewWordsList1.xlsx")
    wl = pd.DataFrame()
    wl['Word'] = tab[col]
    wl = wl.dropna()
    #display(wl)
    wl['Word'] = wl['Word'].str.normalize('NFKD').str.encode('ascii', errors='ignore').str.decode('utf-8')

    wl = [w.lower() for w in wl['Word']]

    ## Création d'une liste contenant les mots dont la catégorie est Désordre
    words_list1 = []
    words_list2 = []
    s = '(?<=\s)'
    ss = '^'
    sss = "(?<=\')"
    i = 0
    for w in wl:
        words_list1.append(s+'['+w[0].upper()+w[0]+']'+w[1:len(w)])
        words_list1.append(ss+'['+w[0].upper()+w[0]+']'+w[1:len(w)])
        words_list1.append(sss+'['+w[0].upper()+w[0]+']'+w[1:len(w)])
        
        words_list2.append(ss+'['+w[0].upper()+w[0]+']'+w[1:len(w)])
    '''
    for elem in appendlist:
        words_list1.append(s+elem)
        words_list1.append(ss+elem)
        words_list1.append(sss+elem)
        words_list2.append(ss+elem)
    '''  
    #print(words_list1)    
        
    return words_list1, words_list2

## Fonction de similarité entre les phrases    
def similar(a, b):
    return SequenceMatcher(None, a, b).ratio()

def similarites(df, col):
    for i in range(len(df[col])):
        for j in range(len(df[col])):
            if i != j and (df[col][i] != None and df[col][j] != None):
                if similar(df[col][i], df[col][j]) > 0.95:
                    #print(similar(df[col][i], df[col][j]))
                    df[col][i] = None
    
    df = df.dropna()
    df = df.reset_index()
    df.drop('index', axis=1, inplace=True)
    
    return df

def guillotine(df):
    _, words_list_type = listeMotsImportants('Type')
    
    df['Info'] = None
    idx = 0
    compt = 0
    for elem in df['Description']:
        patts = ['^[Dd][a-z]+\W+ndeg+\W+','^D[0-9]+\W+','^(?i)[Dd]ommage+\W+[0-9]+','^(?i)[Dd]esordre+\W+[0-9]+']#'^[0-9]+/+\s']
        if multi_re_find(patts,elem) != None:
            df['Info'][idx] = 'Yes'
            compt += 1
        idx += 1
    
    idx = 0
    if compt == 0:
        for elem in df['Description']:
            patts = ['^(?i)[Dd]ommage+\W+','\s+(?i)Dommage+\W+','^[0-9]+(?!.\d)+\W+-','^o+\s']
            if multi_re_find(patts,elem) != None:
                df['Info'][idx] = 'Yes'
                print('Yes')
                compt += 1
            idx += 1
    
    idx = 0
    if compt == 0:
        for elem in df['Description']:
            patts = words_list_type
            if multi_re_find(patts,elem) != None:
                df['Info'][idx] = 'Yes'
                print('YesYes')
                compt += 1
            idx += 1
    
    #display(df)
    
    if compt != 0:
        df = df.dropna()
        df = df.reset_index()
        df.drop('index', axis=1, inplace=True)
    
    df.drop('Info', axis=1, inplace=True)
    
    df['Info2'] = None
    idx = 0
    for elem in df['Description']:
        patts = ['^[Dd][a-z]+\W+ndeg+\W+[0-9]+','^D[0-9]+\W+','^(?i)[Dd]ommage+\W+[0-9]+','^[Dd]esordre+\W+[0-9]+','^[0-9]+(?!.\d)+\W+-','^o+\s','^[0-9]+/+\s']
        pat = multi_re_findPatt(patts, elem)
        if pat != None:
            do = re.findall(pat, elem)[0]
            df['Info2'][idx] = do
        else:
            df['Info2'][idx] = 'None'+str(idx)
            
        idx += 1

    #display(df)
   
    df = df.drop_duplicates('Info2')
    df.drop('Info2', axis=1, inplace=True)
    
    #display(df)
    
    return df

## Classification des désordres
def classificationDésordres(True_Desordres, col):
    Liste1 = []
    Liste2 = []

    for elem in True_Desordres[col]:
        Liste2 = []
        if multi_re_find(['[Aa]ction','[Nn]ecessite', '[Ee]n mesure', '(?<=\s)[Rr]epar', '[Cc]ombler'],elem) != None:
            Liste2 = ['Action', elem]
        elif multi_re_find(['[Dd]esordre','[Pp]robleme','D[0-9]','[Dd]ommage'],elem) != None:
            Liste2 = ['Désordre', elem]
        else:
            Liste2 = ['Désordre', elem]

        Liste1.append(Liste2)

    return Liste1

def rechercheMotsCourants(Desordres,col):
    LL = []
    
    words_list_type,_ = listeMotsImportants('Type')
    words_list_type2,_ = listeMotsImportants('Type 2')
    words_list_objet,_ = listeMotsImportants('Objet')
    words_list_localisation,_ = listeMotsImportants('Localisation')
    patts = ['[Dd][a-z]+\W+ndeg+\W+','D[0-9]+\W+','[Dd]ommage+\W+[0-9]','[Dd]esordre+\W+[0-9]']
            
    for elem in Desordres[col]:
        if multi_re_find(words_list_type,elem) != None:        
            if multi_re_find(words_list_objet,elem) != None:
                LL.append(multi_re_find(words_list_objet,elem))
            else:
                if multi_re_find(words_list_localisation,elem) != None:
                    LL.append(multi_re_find(words_list_localisation,elem))
                else:
                    if multi_re_find(patts,elem) != None:
                        LL.append(multi_re_find(patts,elem))  
        else:
            if multi_re_find(words_list_type2,elem) != None:        
                if multi_re_find(words_list_objet,elem) != None:
                    LL.append(multi_re_find(words_list_objet,elem))
            
    True_Desordres = pd.DataFrame(LL, columns = [col])
    True_Desordres = True_Desordres.drop_duplicates().reset_index().drop('index', axis=1)
    
    return True_Desordres

## Application de la fonction de recherche pour les mots désordre et point
def rechercheDésordres(data):
    i = 0
    liste = ['[Dd]esordre', 'D[0-9]+\s', '(?i)[Dd]ommage','\s+[0-9]+/+\s'] #,'d'+'\d+' ,'[Pp]oint'
    liste2 =['[Pp]robleme', 'Sinistre']
    L = []
    
    #display(data)
    
    ii = 0
    for elem in data['Text']:
        if multi_re_find(liste,elem) != None:
            L.append(unidecode.unidecode(multi_re_find(liste,elem)))
            if len(re.findall('d[ée]sordres? suivants?\s?$',elem)) > 0:
                L.append(data['Text'][ii+1])
                jj = 2
                while len(re.findall('^-',data['Text'][ii+jj])) > 0:
                    L.append(data['Text'][ii+jj])
                    jj += 1
                    
        ii += 1
    
    if len(L) == 0:
        for elem in data['Text']:
            if multi_re_find(liste2,elem) != None:
                L.append(unidecode.unidecode(multi_re_find(liste2,elem)))
        
    Desordres = pd.DataFrame(L, columns = ['Phrase'])
    
    
    #display(Desordres)
    ## Recherche des mots de la liste parmi les descriptions des désordres précédemment trouvés
    
    True_Desordres = rechercheMotsCourants(Desordres, 'Phrase')
    #display(True_Desordres)
    
    noted_idx = []
    idx = 0
    pl = []
    patterns = [['\s+[Dd][a-z]+\W+ndeg','\s(?=[Dd][a-z]+\W+ndeg+[\W+\w]+)','ndeg'],
                ['\s+D[0-9]+\W+','\s(?=D[0-9]+\W+[\W+\w]+)','D[0-9]+\W+'],
                ['\s+(?i)[Dd]ommage+\W+[0-9]+\s','\s(?=(?i)[Dd]ommage+\W+[0-9]+\W+[\W+\w]+)','(?i)[Dd]ommage+\W+[0-9]+\W+'],
                ['\s+[0-9](?!.\d)+\W+-','\s(?=[0-9](?!.\d)+\W+-+\W+[\W+\w]+)','[0-9](?!.\d)+\W+-'],
                ['\s+o+\s',',\s(?=o+\s+[\W+\w]+)','o+\s']]
    
    #display(True_Desordres)
    
    for elem in True_Desordres['Phrase']:
        # D... ndeg
        check = 0
        if re.findall('\s+[Dd][a-z]+\W+ndeg', elem):
            check += 1
            pf = re.split('\s(?=[Dd][a-z]+\W+ndeg+[\W+\w]+)', elem)
            for e in pf:
                if 'ndeg' in e:
                    pl.append(e)
        # D1...
        if re.findall('\s+D[0-9]+\W+', elem):
            check += 1
            pf = re.split('\s(?=D[0-9]+\W+[\W+\w]+)', elem)
            for e in pf:
                if re.findall('D[0-9]+\W+', e):
                    pl.append(e)

        # Dommage 1..
        if re.findall('\s+(?i)[Dd]ommage+\W+[0-9]+\s', elem):
            check += 1
            pf = re.split('\s(?=(?i)[Dd]ommage+\W+[0-9]+\W+[\W+\w]+)', elem)
            for e in pf:
                if re.findall('(?i)[Dd]ommage+\W+[0-9]+\W+', e):
                    pl.append(e)

        # - 1..
        if re.findall('\s+[0-9](?!.\d)+\W+-', elem):
            check += 1
            pf = re.split('\s(?=[0-9]+(?!.\d)+\W+-+\W+[\W+\w]+)', elem)
            for e in pf:
                if re.findall('[0-9]+(?!.\d)+\W+-', e):
                    pl.append(e)

        # o ..
        if re.findall('\s+o+\s', elem):
            check += 1
            pf = re.split('\s(?=o+\s+[\W+\w]+)', elem)
            for e in pf:
                if re.findall('o+\s', e):
                    pl.append(e)
                    
        # o ..
        if re.findall('\s+[0-9]+/+\s+', elem):
            check += 1
            pf = re.split('\s(?=[0-9]+/+\s+[\W+\w]+)', elem)
            for e in pf:
                if re.findall('[0-9]+/+\s', e):
                    pl.append(e)
        
        if check > 0:
            noted_idx.append(idx)        
        #print(idx)
        idx += 1
        
    ajoutDesordres = pd.DataFrame(pl, columns = ['Phrase'])
    #display(ajoutDesordres)
    #display(noted_idx)
    
    
    for i in noted_idx:
        True_Desordres = True_Desordres.drop(i, axis = 0)
    '''
    if len(ajoutDesordres) > 0: 
        True_Desordres = ajoutDesordres
    '''
    True_Desordres = pd.concat([True_Desordres, ajoutDesordres], axis = 0).reset_index().drop('index', axis = 1)
    #display(True_Desordres)
    ## Nettoyage des descriptions
    idx = 0

    for l in True_Desordres['Phrase']:
        if len(re.findall('D[0-9]+',l)) > 0:
            True_Desordres['Phrase'][idx] = re.findall('(?=D[0-9]+)\w+[\w+\W+]+[\d+\D+]', l)
            True_Desordres['Phrase'][idx] = ''.join(True_Desordres['Phrase'][idx])
        idx += 1

    #Liste1 = classificationDésordres(True_Desordres)
    
    DnA = True_Desordres.rename(columns = {"Phrase" : 'Description'})
    DnA = similarites(DnA, 'Description')
    #display(DnA)
    ## Si aucun désordre n'est trouvé
    LLL = []

    if len(DnA) == 0:
        D = rechercheMotsCourants(data,'Text')
        D = D.rename(columns = {'Text': 'Phrase'})
        D = similarites(D, 'Phrase')       
        
        Liste2 = classificationDésordres(D, 'Phrase')
    
        DnA = pd.DataFrame(Liste2, columns = ['Type','Description'])
    
    DnA = guillotine(DnA)
    DnA = rechercheMotsCourants(DnA, 'Description')
    
    Liste1 = classificationDésordres(DnA, 'Description')
    DnA = pd.DataFrame(Liste1, columns = ['Type','Description'])
    
    #display(DnA)
    return DnA

def Désordres(fichier):
    print(fichier)
    data = extractionPDF(fichier)
    #words1, words2 = listeMotsImportants('Type')
    data = reconstruction(data)
    data = reconstructioEntités(data)
    #display(data)
    data = rechercheDésordres(data)
    
    return data

def tableauDésordres(fichier):
    tab = pd.DataFrame(columns = ['Fichier','Type de Fichier','Type','Description'])
    #listeFichiers = []
    
    #for (repertoire, sousRepertoires, fichiers) in walk(monRepertoire):
    #    listeFichiers.extend(fichiers)
    
    elem = f'wetransfer_rapports_2021-12-06_0911/{fichier}.pdf'
    elem = fichier
    #for elem in listeFichiers: #monRepertoire+'\\'+elem
    tabDésordres = Désordres(elem)
    tabDésordres['Fichier'] = elem
    if re.findall('(?i)pr[ée]li', elem):
        tabDésordres['Type de Fichier'] = 'Rapport préliminaire'
    elif re.findall('(?i)inter', elem):
        tabDésordres['Type de Fichier'] = 'Rapport intermédiaire'
    elif re.findall('(?i)d[ée]f', elem):
        tabDésordres['Type de Fichier'] = 'Rapport définitif'
    elif re.findall('(?i)compl', elem):
        tabDésordres['Type de Fichier'] = 'Rapport complémentaire'
    elif re.findall('(?i)exp', elem) or re.findall('(?i)audi', elem):
        tabDésordres['Type de Fichier'] = "Rapport expertise/audit"
    elif re.findall('(?i)convo', elem):
        tabDésordres['Type de Fichier'] = 'Convocation'
    elif re.findall('(?i)note', elem):
        tabDésordres['Type de Fichier'] = 'Note technique/information'
    elif re.findall('(?i)ass', elem):
        tabDésordres['Type de Fichier'] = 'Assignation'
    else:
        tabDésordres['Type de Fichier'] = 'Autre'
    #print(tabDésordres['Type de Fichier'])
    tab = pd.concat([tab,tabDésordres])
    
    tab = tab.reset_index()
    tab = tab.drop('index', axis=1)
    #tab = tab.to_json()
    
    return tab


# In[6]:


#monRepertoire = 'wetransfer_mise-en-cause_2021-10-12_1000\Mise en cause\D.O'
#monRepertoire = 'wetransfer_rapports_2021-12-06_0911' #'test'
#monRepertoire = 'RappPreli' 
#monRepertoire = 'RappDef'
#monRepertoire = 'DésordresHédi' 
#monRepertoire = 'DursATraiter' 
#monRepertoire = 'RapportTest'
#monRepertoire = 'RapportARegarder'
#monRepertoire = 'wetransfer_a10200035-do-av1-rapport-complementaire-20220228_094613-pdf_2022-02-28_1314'
#fichier = 'A-12200206-C Rapport préliminaire+-+20200720'

#tab = tableauDésordres(fichier)
#display(tab)


# In[8]:


#jsondf = tab.to_json()
#jsondf


# In[68]:


#taaaab = pd.DataFrame(pd.concat([tab['Fichier'],tab['Description']], axis = 1))


# In[69]:


#taaaab


# In[7]:


#uni = pd.concat([tab['Fichier'],tab['Type de Fichier']], axis = 1)
#uni = pd.DataFrame(uni.drop_duplicates())

#display(uni)


# In[8]:


#uni.to_excel('ListeFichiers2.xlsx')


# In[9]:


#tab.to_excel('Désordres62.v4.xlsx')


# In[10]:


#tab


# In[11]:


#tabpreli = tab[tab['Type de Fichier'] == 'Rapport préliminaire']
#tabpreli


# In[12]:


#tab.to_excel('DesordresàClassifier.xlsx')

