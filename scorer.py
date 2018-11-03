# Team name: Evil Geniuses
# Authors:
# Vijayasaradhi Muthavarapu (Vijay)
# Nikhil Reddy Pathuri (Nikhil)
# Dilip Molugu (Dilip)
# Date: 10/17/2018

#1. This program helps in calculating the Accuracy and Confusion Matrix for the predictions we made using the tagger.py by comparing its results to the Gold Standard file.

#2. Inorder to make this program work first we need to install the required libraries. Then in the command line arguments you need to pass a minimum of 3 arguments. 
#   -> The first argument should be the test results data obtained from executing tagger.py.
#   -> The second argument should be the Gold Standard data with correctly tagged words.
#   -> The third argument is the standard output file to which our program prints the results to.

# An example to run this program:
# >python scorer.py pos-test-with-tags.txt pos-test-key.txt > pos-tagging-report.txt
# Sample Output:
# Accuracy = 0.8649314538127166
#           /#   /$  /''  /(  /)    /,    
# /#         5    0    0   0   0     0 
# /$         0  375    0   0   0     0 
# /''        0    0  531   0   0     0 
# /(         0    0    0  76   0     0

#3. About our code and algorithm:
# Step 1: It reads two text files from the arguments and preprocess the data by removing few unwanted characters like '\n','[',']' from the text.
# Step 2: We tokenize tags from words using regular expressions and convert the data into a 2d list.
# Step 3: Combine the Predicted tags with Gold Standard tags inorder to compare and calculate the Accuracy of the tagger.py program.
# Step 4: Create the Confusion Matrix using the confusion_matrix() function from sklearn.metrics package and print the results to the standard output.

import sys
import re
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix

def pre_processing(arg):
    train = open(sys.argv[arg], "r").read().rstrip().replace('\n', '').replace('[', ''). replace(']', '')
    train1 = re.sub(' +',' ',train)
    return train1

def create_table(arg,name):
    results = arg.strip().split(" ")
    text=[]
    for row in results:
        row=re.sub(r'(\/[A-Z]*(\|?)[^\d|^/]+)$',r' \1',row)
        text.append(row.strip().split(" "))
        
    cols = ["words",name]
    df = pd.DataFrame(data=text, columns=cols)
    return df


result_tags = pre_processing(1)
key_tags = pre_processing(2)

results_df = create_table(result_tags,"result_tags")
key_df = create_table(key_tags,"key_tags")

output_df = pd.concat([results_df, key_df["key_tags"]], axis=1)

#calculate Accuracy
count=0
output_list = output_df.values.tolist()
for i in range(len(output_list)):
    if(output_list[i][1]==output_list[i][2]):
        count+=1
sys.stdout.write("Accuracy = "+str(count/len(output_list))+"\n")

c= confusion_matrix(output_df["key_tags"],output_df["result_tags"])
c= pd.DataFrame(c,index=np.sort(output_df["key_tags"].unique()).tolist(),columns=np.sort(output_df["key_tags"].unique()).tolist())

sys.stdout.write(c.to_string())