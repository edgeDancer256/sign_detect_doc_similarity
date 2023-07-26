import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.filterwarnings('ignore')

import string
from gensim.parsing.preprocessing import remove_stopwords

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

#Loading the model
model = load_model('./model_sign_classifier.h5', compile=False)
#Processing the input image
#Team 15
#test_image = image.load_img('./Trial/PES1PG22CS046.jpg', target_size = (64,64))
#test_image = image.load_img('./Trial/PES1PG22CS011.jpg', target_size = (64,64))
#Team 18
#test_image = image.load_img('./Trial/PES1PG22CS013.jpg', target_size = (64,64))
#test_image = image.load_img('./Trial/PES1PG22CS025.jpg', target_size = (64,64))
#False Positive
test_image = image.load_img('./Trial/haseeb1.jpg', target_size = (64,64))
test_image = image.img_to_array(test_image)
test_image=test_image/255
test_image = np.expand_dims(test_image, axis = 0)
#Predicting the class for the input image
result = model.predict(test_image)

srn = ''
if(np.argmax(result[0]) < 10):
    srn = '0'+str(np.argmax(result[0]) + 1).strip()
else:
    srn = str(np.argmax(result[0]) + 1).strip()

res = 'This signature belongs to PES1PG22CS0' + srn

#Printing the predicted class
print(res)

#Utility functions to pre-process the documents
def pre_process(doc):
    temp = doc.lower()
    temp = temp.translate(str.maketrans('', '', string.punctuation + string.digits))
    temp = remove_stopwords(temp)

    return temp

docs = dict()
for file_name in os.listdir('../Dataset/TXTs'):
    f = open('../Dataset/TXTs/'+file_name, 'r')
    temp = f.read()
    temp = pre_process(temp)
    docs[file_name] = temp
    f.close()

#Function to calculate the Jaccard Similarity index between the documents
def jaccard_similarity(doc1, doc2):
    word_1 = set(doc1.split())
    word_2 = set(doc2.split())

    inter = word_1.intersection(word_2)
    union = word_1.union(word_2)

    return float(len(inter) / len(union))

#Calculating and displaying the similarity index as 1 versus all
for key in docs.keys():
    print('Similarity of File : PES1PG22CS0' + srn +'.pdf with File : '+ key[:-4] + '.pdf is: ' + str(jaccard_similarity(docs['PES1PG22CS0'+srn+'.txt'], docs[key])))

#Displaying the PDF corresponding to the predicted class
import webbrowser
webbrowser.open_new_tab('../Dataset/PDFs/PES1PG22CS0'+ srn +'.pdf')