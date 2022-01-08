![title](https://msnews.github.io/assets/img/icons/logo.png)
***
# MIND - Deep Learning Recommendation System
***
![python](https://img.shields.io/badge/Python-v3.7.6-blue) ![pytorch](https://img.shields.io/badge/pytorch-v1.10-red)
![version](https://img.shields.io/badge/Version-v1.0-green)
***
> **Authors:**
> - [Guy Dahan](https://github.com/Guydada)
> - [Guy Krothammer](https://github.com/guykrot)
***
> **Submission:**
> - Instructor: Dr. Liron Izhaki Allerhand
> - Course: [Introduction to Search, Information Retrieval and Recommender Systems](https://www.ims.tau.ac.il/tal/syllabus/Syllabus_L.aspx?lang=EN&course=0560416201&year=2021)
> - Semester: October 2021
> - Course ID: 0560416201
> - Faculty: [Engineering](https://en-engineering.tau.ac.il/)
> - Department: [Digital Sciences for Hi-Tech](https://en-engineering.tau.ac.il/BSc-in-Digital-Sciences-for-Hi-Tech)
> - [Tel-Aviv University](https://english.tau.ac.il/)
***
![Tel-Aviv University](https://english.tau.ac.il/sites/default/files/TAU_Logo_HomePage_Eng.png)

## Requirements
- Python 3.7.6
- PyTorch 1.10
- Anaconda3
- MIND-Dataset (small or full)

## Installation
Clone the repository, and run the following command (in the root directory):
```
$ conda create --name <env> --file requirements.txt
```
## Usage
After installation, you can run the following command to activate the environment:
```
$ conda activate <env>
```
Then you can run the following command to run the code:
```
$ python main.py
```
For a full list of commands, run:
```
$ python main.py --help
```
After running the code, a prompt will appear asking for mode, choose between:
- 'tfidf' - TF-IDF mode
- 'model' - Model mode

## Abstract
This project is a final submission project in the course: "Introduction to Search,
Information Retrieval and Recommender Systems".The dataset used is the [MIND](https://msnews.github.io/)
dataset by Microsoft. The dataset contains articles from the [Microsoft News](https://www.microsoft.com/en-us/news/) 
and [Microsoft Blog](https://blogs.msdn.microsoft.com/) websites. In addition, the behavior of users (over 2.5 million
users) is also included. The dataset is divided into train, validation and test sets - we will discuss the matter of this
division in the next sections.  
<br>
Furthermore, a "SMALL-MIND" dataset was supplied as well, we chose to use it as our main set for training and testing 
due to time and computational constraints.  
<br>
Our main challenge was figuring out how to represent large amount of text data in a numerical form. We used TF-IDF 
due to its simplicity and efficiency, applying 2 main approaches for recommendation:
<br>
- *A clean TF-IDF approach - Using cosine similarity* 
- *A hybrid approach - using TF-IDF, content based filtering and collaborative filtering*
 <br>

In both approaches, we defined the baseline for the recommendations - the most popular articles by clicks.
We have implemented different metrics for evaluating the quality of our recommendations, mostly we chose to use
nDCG score in order to have a comparable result the existing MIND projects. 

### Coding Standards
The code is written in Python 3.7.6, and is divided into the following modules:
- `main.py` - The main file, which contains the main function.
- `mind.py` - The file that contains the MIND dataset classes.
- `models.net_models.py` - The file that contains the neural network models.
- `models.utils` - The file that contains the utility functions - `tensorize.py`, `load_preprocess.py`, `evaluate.py`.

We have taken a big effort to try and withstand the following:
- [PEP8](https://www.python.org/dev/peps/pep-0008/)
- OOP principles
- Documentation
- Simplicity
- Minimal code duplication using inheritance and composition


## Define the Problem
The problem we are trying to solve is to recommend articles to users based on their behavior and the articles they 
have read and interacted with. This seemingly simple problem is actually a huge challenge in the field of recommendation.
We think this dataset is especially interesting because it gives us the opportunity to explore both collaborative and
content based filtering approaches.

## Data
### News Articles
First, let's see some basic exploration of the data. The categories are:
![categories](images/categories.png)

The data itself (referring as mentioned to SMALL-MIND) contains 51282 unique articles, including their categories, subcategories, abstracts, and content. For
a full review of the data's structure, please refer to the
[README](https://github.com/msnews/msnews.github.io/blob/master/assets/doc/introduction.md) in Microsoft News repository.

### Behaviors 
The data contains the behavior of users (over 2.5 million users) and the articles they have read and interacted with. The
SMALL-MIND contains 50,000 unique users who interacted only with articles that appear in the MIND-SMALL news dataset.

### Undersampling
We noticed early on that most users just don't interact with articles. Therefore, we decided to undersample the users 
behaviors. 

## Pure TF-IDF

### Performance Metrics

***
## Deep Learning Approach

### Performance Metrics



### References
- [MIND](https://msnews.github.io/), by Microsoft
- [Microsoft Recommender Repository](https://github.com/microsoft/recommenders)
- Stevens, E., Antiga, L., & Viehmann, T. (2020).
[Deep Learning With PyTorch](https://www.google.com/search?client=firefox-b-e&q=deep+learning+with+pytorch+). Manning Publications.



