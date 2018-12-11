# POS-tagging

# Team name: Evil Geniuses
# Authors:
# Vijayasaradhi Muthavarapu (Vijay)
# Nikhil Reddy Pathuri (Nikhil)
# Dilip Molugu (Dilip)
# Date: 10/17/2018

1. This program helps in performing POS tagging for an untagged data. It takes tagged data as input for training the model and an untagged data on which it predicts the POS tags.

2. Inorder to make this program work first we need to install the required libraries. Then in the command line arguments you need to pass a minimum of 3 arguments. 
   -> The first argument should be the training data with tagged words.
   -> The second argument should be the testing data with untagged words.
   -> The third argument is the standard output file to which our program prints the results to.


 An example to run this program:
 >python tagger.py pos-train.txt pos-test.txt > pos-test-with-tags.txt

Sample Output:

No/RB ,/, it/PRP was/VBD n't/RB Black/NNP Monday/NNP ./. But/CC while/IN the/DT....

3. About our code and algorithm:
 Training data: We used the text file provided to us with pre-tagged data for training.
 Program Logic: 
 Step 1: First our program reads train and test text files and appends the files into a variable. We also remove few unwanted characters like '\n','[',']' from the text.
 Step 2: We tokenize tags from words using regular expressions and convert the data into a 2d list.
 Step 3: Calculate the frequency distribution of the tags.
 Step 4: Calculate the frequency distribution of a tag corresponding to a word using the groupby command.
 Step 5: Calculate the frequency distributoin of tag with its previous tag.
 Step 6: Clean and tokenize the test data for prediction.
 Step 7: Predict the tags by calculating most likely probabilites of words and tags and also tags with previous tags. The Base accuracy = 85.56%
 Step 8: Apply rules on the predictions: 
    1) #rule1 : if tag is '/NNPS' make it '/NNP' --> Accuracy = 85.63%
    2) : if word is int make tag to /CD --> Accuracy = 86.39%
    3) if prev is (/NN or /NNS) current is (/IN with /WDT) next is (/VBD or /VBZ or /VBP) --> Accuracy = 86.47%
    4) words with $ will have /$ tag --> Accuracy = 86.48%
    5) words with ( will have /( tag --> Accuracy = 86.49%

 Step 9: Combine the predicted tags with words and print to the standard output file.
 Key Feature:We used the formula: P(wi/ti)*P(ti/ti-1) for calculating the likelihood.
