#!pip install pdfminer.six
#!pip install streamlit
#!pip install wordcloud
#!pip install docx2txt
#!pip install nltk
#!pip install sklearn
#!pip install tkinter

import nltk
nltk.download("stopwords")
nltk.download("punkt")
import streamlit as st
import io
from pdfminer.converter import TextConverter
from pdfminer.pdfinterp import PDFPageInterpreter
from pdfminer.pdfinterp import PDFResourceManager
from pdfminer.pdfpage import PDFPage
 #Docx resume
import docx2txt
 #Wordcloud
import re
import operator
from nltk.tokenize import word_tokenize 
from nltk.corpus import stopwords
set(stopwords.words('english'))
from wordcloud import WordCloud
from nltk.probability import FreqDist
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from pdfminer.high_level import extract_text
import tk
import os
import numpy as np
from cachetools import TTLCache

st.title("Resume Screening")
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/thumb/7/73/Applus%2B_IDIADA_Logo.svg/1200px-Applus%2B_IDIADA_Logo.svg.png", width = 180)
root = tk.Tk()
root.withdraw()

# Make folder picker dialog appear on top of other windows
root.wm_attributes('-topmost', 1)
CV_loc_1 = ""
JD_loc = ""
# Folder picker button


st.sidebar.write("---")


from tkinter import *
from tkinter import filedialog





#select_JD = st.sidebar.checkbox('Select JD')
#select_final_folder = st.sidebar.checkbox('Select Output Folder')
score = st.sidebar.number_input("Percent Accuracy required", min_value = 0, step = 1)
st.sidebar.write("---")
num_CV_req = st.sidebar.number_input("Number of top CVs required", min_value = 0, step = 1)

#select_CV_folder = st.sidebar.radio('Select CV Folder',('Select','Dont select'))
st.sidebar.write("---")
   


    

button = st.sidebar.button("Select Folders and JD")

if button:
    CV_loc = st.sidebar.text_input('Selected CV folder path:', filedialog.askdirectory(master = root, title ="Select CV Folder"))
    
    file = filedialog.askopenfile(master = root, title ="Select JD File")
    if file:
        JD_loc = st.sidebar.text_input('Selected JD file path:', os.path.abspath(file.name))

    
    final_loc = st.sidebar.text_input('Selected Destination folder path:', filedialog.askdirectory(master = root, title ="Select Destination Folder"))
    
    def read_word_resume(word_doc):
        resume = docx2txt.process(word_doc)
        resume = str(resume)
        #print(resume)
        text =  ''.join(resume)
        text = text.replace("\n", "")
        if text:
            return text


    # In[4]:


    ## Delete more stop words
    #if you want a certain word  to not appear in the word cloud, add it to other_stop_word
    other_stop_words = ['junior', 'senior','experience','etc','job','work','company','technique',
                        'candidate','skill','skills','language','menu','inc','new','plus','years',
                       'technology','organization','ceo','cto','account','manager','data','scientist','mobile',
                        'developer','product','revenue','strong','impact','ability','lower','cae','vehicle','good','problems',
                       'global','seat','speed']

    def clean_job_decsription(jd):
             ## Clean the Text
             # Lower
        clean_jd = jd.lower()
             # remove punctuation
        clean_jd = re.sub(r'[^\w\s]', '', clean_jd)
             # remove trailing spaces
        clean_jd = clean_jd.strip()
             # remove numbers
        clean_jd = re.sub('[0-9]+', '', clean_jd)
             # tokenize 
        clean_jd = word_tokenize(clean_jd)
             # remove stop words
        stop = stopwords.words('english')
        clean_jd = [w for w in clean_jd if not w in stop]
        clean_jd = [w for w in clean_jd if not w in other_stop_words]
        return(clean_jd)





    # In[5]:


    def create_word_cloud(jd):
        corpus = jd
        fdist = FreqDist(corpus)
        #print(fdist.most_common(100))
        words = ' '.join(corpus)
        words = words.split()

         # create a empty dictionary  
        data = dict() 
        #  Get frequency for each words where word is the key and the count is the value  
        for word in (words):     
            word = word.lower()     
            data[word] = data.get(word, 0) + 1 
        # Sort the dictionary in reverse order to print first the most used terms
        dict(sorted(data.items(), key=operator.itemgetter(1),reverse=True)) 
        word_cloud = WordCloud(width = 800, height = 800, 
        background_color ='white',max_words = 500) 
        word_cloud.generate_from_frequencies(data) 

         #plot the WordCloud image
        plt.figure(figsize = (10, 8), edgecolor = 'k')
        plt.imshow(word_cloud,interpolation = 'bilinear')  
        plt.axis("off")  
        plt.tight_layout(pad = 0)
        plt.show()


    # In[6]:


    def get_resume_score(text):
        cv = CountVectorizer(stop_words='english')
        count_matrix = cv.fit_transform(text)
        #Print the similarity scores

        #get the match percentage
        matchPercentage = cosine_similarity(count_matrix)[0][1] * 100
        matchPercentage = round(matchPercentage, 2) # round to two decimal

        #print(" resume matches about "+ str(matchPercentage)+ "% of the job description.")
        return matchPercentage


    # In[7]:


    import os
    job_description = extract_text(JD_loc)     #change extract_text to read_word_resume to read a docx file
    clean_jd = clean_job_decsription(job_description) 
    create_word_cloud(clean_jd) 
    score_dict = {}
    path = CV_loc   ##enter location of source folder of cvs
    # Change the directory
    os.chdir(path)
    # iterate through all file
    for file in os.listdir():
        if file.endswith(".pdf"):
            file_path = f"{path}/{file}"
            resume = extract_text(file_path).lower()
            text = [resume, job_description]
            #print(file,': ')
            #get_resume_score(text)
            score_dict[file] = get_resume_score(text)
            #print("\n") 

        elif file.endswith(".docx"):
            file_path = f"{path}/{file}"
            resume = read_word_resume(file_path)
            text = [resume, job_description]
            #print(file,': ')
            #get_resume_score(text)
            score_dict[file] = get_resume_score(text)
            #print("\n")  

        else:
            file_path = f"{path}/{file}"
            st.write(file, "----Not Scanned")
    st.write("---")


    score_dict = dict((k, v) for k, v in score_dict.items() if v >= np.percentile(v,score))
    
    def find_nearest(array, value):
        array = np.asarray(array)
        idx = (np.abs(array - value)).argmin()
        return idx
    score_dict_1={}
    l1 = sorted(list(score_dict.values()), reverse = True)
    temp = np.percentile(l1, 80)
    final_percentile = find_nearest(l1,temp)
    for i in range(0,final_percentile):
        for key,val in score_dict.items():
            if val == l1[i]:
                score_dict_1[key] = val
            


   
    import shutil
    import heapq
    source_folder = CV_loc    ##enter location of source folder of cvs
    destination_folder = final_loc    ##enter location of destination folder of cvs

    #score = int(input("Enter the score threshold: "))
    ## below line will store resumes with a score greater than score
    #score_dict = dict((k, v) for k, v in score_dict.items() if v >= accuracy) ##change the last numeric value to get a score greater than
    i = 0
    for file_name in heapq.nlargest(num_CV_req, score_dict_1, key=score_dict_1.get):   ##change the first attribute to the number of resumes you want to short-list
        # construct full file path
        st.write(file_name)
        source = source_folder + '/' + file_name
        destination = destination_folder + '/' + file_name
        i = i+1
            # copy only files
        if os.path.isfile(source):
            shutil.copy(source, destination)
            os.rename(destination, destination_folder + '/'+ str(i) +'_'+ file_name)
            print('copied', file_name)
    st.success("The Resumes are successfully copied to your folder. Thank You")
