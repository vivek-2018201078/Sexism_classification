Information Retrieval and Extraction
Identification and analysis of Instagram posts
involving Sexist Stereotyping

Major Course Project


ABSTRACT
Sexism, an injustice that subjects women and girls to enormous suffering, manifests in blatant as well as subtle ways. Online sexism has become prevalent in the age of internet. There are more and more records of experiences of sexism on web. Social media platforms seems to be most pertinent among it. We are working on Instagram data to Identify and analyze accounts of sexism through posts. Our secondary task is to provide classification mechanism for identifying if a post is sexist or contains account of sexism in any manner or not. Aim is to try different classification models and compare results.

RELATED WORK
Most of our work is based on readings from paper Multi-label Categorization of Accounts of Sexism using a NeuralFramework. Previous work before that involved hate speech detection and sexism categorization. Most of the papers mentioned there does not directly involves sexism classification task but its variation with different machine learning models. Paper by Jha and Mamidi categorizes tweets as benevolent, hostile or non-sexist using SVM and bi-LSTM  with attention. All of these papers had annotated data readily available or generated manually. Since Instagram dataset is not available related to this subject, we had to come up with our own data retrieval and filtering techniques. 


DATASET 
Unlike Twitter, Instagram’s data retrieval API has many restrictions. From previous work, dataset for instagram is not readily available. Our crucial task was to create relevant dataset. Extracting post information for instagram is only possible by user or by hashtags. For our purpose to get data related to gender bias, we chose hashtag based retrieval. Researching on instagram posts which has frequent mentions of sexism, gender bias and research papers on it, we came up with hashtags which will possibly provide posts which exhibits gender bias.

Hashtags crawled : #everydaysexism, #genderbias, #genderstereotype, #heforshe, #mencallmethings, #metoo, #misogynist, #notallmen, #questionsformen, #slutgate, #wagegap, #weareequal, #womenareinferior, #workplaceharassment, #yesallwomen
Foreach post 100 - 5000 posts were crawled with limitation on caption length set to 10 words.
Posts crawled using above hashtags are labeled as positive data (accounts of sexism present). Random hashtags crawled and posts collected using them labeled as negative data(accounts of sexism absent).

Note - Although hashtag based data can not be definitely classify as positive or negative, it provides baseline distinction between what can be classified as sexist or not. 

DATA PRE-PROCESSING STEPS:
Following Data pre-processing steps are done.
Tokenised text using regex tokeniser. 
Regex used: [a-zA-Z0-9_#]+ 
Removes text written in languages that do not use English alphabet like Hindi, Urdu, Arabic, Chinese, Japanese, etc.
Removes punctuations other than _, # (Kept hashtags as it is)
Removes emojis
Removed numbers from the text
Stopword removal
Lemmatisation
Removed words that are not present in English vocab
Case folding
DATA FILTERING:
Data is filtered using following criteria:
Separate restrictions on length of caption in a post and length of comments in a post. Here, length is the number of words present in pre-processed text.
Total no. of posts in final dataset : 7280 
Ratio of positively labeled data to negatively labeled data is approximately one. To reduce irrelevant posts from positively labeled data, k-means clustering was performed with 2 clusters. 
WORD EMBEDDINGS:
Multiple word embedding algorithms used to get vector representations for caption text. 
Word embedding of a post is calculated by taking the average of word embeddings of all the words in the post.
Following word embedding algorithms are used:
Word2Vec : 
pre-trained using google news corpus containing (3 billion running words) word vector model (3 million 300-dimension English word vectors). 
Vector size : 300
Glove : 
pre-trained on twitter dataset provided by Stanford. 
Vector size : 200
All approaches tried after this are used on results of both w2v and glove embedding model.

BASELINE APPROACHES:
 From the above retrieved dataset, our Classification task falls into two types of
approaches:
○ Unsupervised Methods.
○ Supervised Methods.

UNSUPERVISED METHOD:
Since the data is based on pre trained word embeddings and creating vector for every post, we could assume that each vector carries semantic meaning. We applied clustering approach to divide the dataset into 2 different clusters using K-Means clustering algorithm with intention to divide posts which carry similar semantic structure. Since dataset contains both positive and negative samples, it is assumed that they will be in different clusters. Results are discussed below. 
Purity of clusters observed in K-Means clustering algorithm : 52.5%



SUPERVISED METHODS:
    For supervised approaches, extracted posts with selected hashtags are given positive label and posts with dummy hashtags are given negative labels. After applying word embedding on combined data with 300 dimension vector for caption in each post, we tried the following approaches:
Logistic Regression: Applied standard Logistic Regression algorithm with 80 - 20 Split on train to test   data.
Naive Bayes: Applied standard Naive bayes algorithm with 80 - 20 Split on train to test data.
Random Forest: Random Forest Ensemble implemented with 100 decision trees.
SVM: Applied Support vector machines with linear kernel and the regularization factor 1.



ARCHITECTURE






DEEP NEURAL NETWORK and LSTM:
Following configuration is used for both deep neural network and LSTM models:
Input Layer : word embeddings of instagram post 
Hidden Layer : Two hidden layers where number of neurons is equal to half of vector size of word embedding used at the input layer and “relu” activation function is used.
Output Layer : One neuron with “sigmoid” activation function. Threshold value 0.5 is used to label the post.
“Random normal” kernel initializer and “binary cross-entropy” loss function is used

Hyper-Parameter tuning using cross-validation:
For both Deep Neural Network and LSTM models, 3-fold cross validation is performed with                  training data only and below parameter values were varied :
Size of batch (64, 128)
No. of epochs (50, 100, 150)
Type of optimizers (adam, rmsprop)

Cross-Validation results:
Best results observed with below parameter configuration:
Deep NN :
Word2Vec : rmsprop optimizer, 128 batch size, 50 epochs.
Glove: rmsprop optimizer, 128 batch size, 150 epochs
LSTM : 
Word2Vec : 
Glove: 


RESULTS:
Accuracy comparison of different models with different word-embeddings:
MODEL
WORD EMBEDDING
ACCURACY
Naive Bayes
Word2Vec
73.00
Glove
77.00
Logistic Regression
Word2Vec
83.00
Glove
81.00
Random Forest
Word2Vec
87.63
Glove
87.50
Support Vector Machine
Word2Vec
83.00
Glove
80.00
Deep Neural Network
Word2Vec
85.93
Glove
86.19
LSTM
Word2Vec


Glove



ANALYSIS:
GITHUB LINK: https://github.com/karumugamio/IREProjectGroup12
WEBPAGE LINK: 

