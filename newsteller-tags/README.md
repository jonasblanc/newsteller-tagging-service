# newsteller-tags
Juypter notebooks used to train [fasttext](https://fasttext.cc) ML model to classify articles into categories.

The big picture: Some newspapers classify their articles into categories. We are able to recover the classification be observing urls. Thus we can train the model to classify articles with supervised data.

## Table of contents
  * [ClasiffyArticlesSaveInOneFile](ClasiffyArticlesSaveInOneFile.ipynb) is used to classify multiple articles and save them into a single Json file that can be read by the mobile app
  * [CrossValidation](CrossValidation.ipynb) is used to validate the hyperparameters of the model (see [CrossValidation](#Cross-Validation)).
  * [LoadModelClassifyArticle](LoadModelClassifyArticle.ipynb) loads a pre-trained model and classifies an article. This procedure is used in [microservices](../microservices) to classify article on the fly.
  * [TopWordsPerLabel](TopWordsPerLabel.ipynb) computes the closest words to each label with help of a trained model.
  * [TrainSaveModel](TrainSaveModel.ipynb) trains a model and saves it. This procedure is used in [microservices](../microservices) to keep the model up to date.
  * [WordsVisualisation](WordsVisualisation.ipynb) creates a 2D visualisation of the distance between words and labels.

## Cross-Validation
All tests were done using articles in french. In this section we are trying to find values of hyperparameters giving the best accuracy.
  * lr - the learning rate
  * text size - the maximum number of words per article
  * epoch - the number of times the model goes over the data
  * dimension - the size of vectors representing the labels and the words.
  
In brief:
 * We get better accuracy using title and text than just text.
 * We don't get much improvement using pretrained vectors.
 * The number of words per considered article has a large influence on accuracy. The best value varies between trials but it seems to stay around 1000.
  
I'll now explain how I came to the conclusions above. First, I noticed that even with a small learning rate the model learns pretty fast.  

<img src="https://user-images.githubusercontent.com/44334351/97585393-63644980-19f9-11eb-9cdc-96e20ec3e6ea.png" width="400">

We can see that with a learning rate of 0.2 both validation and test accuracy stabilise after a few epochs

Then I looked at the dimension and text size. The resulting heatmap looks as follows:  
<img src="https://user-images.githubusercontent.com/44334351/97585358-5b0c0e80-19f9-11eb-8fcf-6f5b08ae2e81.png" width="400">  

It is interesting to note that the biggest dimension and largest number of words by articles are not the ones getting the best accuracy as one could expect. I chose the categories arbitrarily, sciences category is the one with fewest articles in training, validation and test sets. It seems that the model has more difficulties to classify them. The text size has a bigger influence on the accuracy than the vector size.

Until now when I pre-process the articles removing stop words etc, for each article I concatenate the title with the text. I did the same process as above but this time based only on the text. The resulting heatmap looks as follows:  
<img src="https://user-images.githubusercontent.com/44334351/97585422-6c551b00-19f9-11eb-9c35-73bda1802f8f.png" width="400">  
  
| Accuracy       | Number of articles | With title | Without title |
|----------------|--------------------|------------|---------------|
| Validation set | 2100               | 0.9174     | 0.9193        |
| Total test set | 2666               | 0.9283     | 0.9238        |
| sciences       | 249                | 0.7670     | 0.7510        |
| economie       | 1101               | 0.9600     | 0.9573        |
| sport          | 321                | 0.9626     | 0.9626        |
| culture        | 395                | 0.9139     | 0.9139        |
| sante          | 600                | 0.9283     | 0.9200        |

There is no fundamental change, the heatmap is overall darker, meaning that the model globally performs better with than without title. 

Finally, I tried to train a model using [pretrained vector](https://fasttext.cc/docs/en/crawl-vectors.html) trained by fasttext. The result is equivalent:

| Accuracy | With pretrained vectors | Without pretrained vector |
|----------|-------------------------|---------------------------|
| Total    | 0.9343                  | 0.9276                    |
| sciences | 0.7163                  | 0.6778                    |
| economie | 0.9536                  | 0.9607                    |
| sport    | 0.9719                  | 0.9750                    |
| culture  | 0.9221                  | 0.9041                    |
| sante    | 0.9525                  | 0.9367                    |

To sum up, the model learns in few epochs, we need to find meaningful and well balanced categories. Titles provide some supplementary knowledge. Pretrained vectors improve the result a little bit, but come with a cost of 4.5 GB (size of pretrained vectors).

## Data visualisation and distances
The ML model uses vectors to represent words and labels. Thus we are able to exploit this notion of distance between labels/words to learn from the vector space built by the model.

We can for example find the ten closest words per label. Below are the top ten words per label (it's never to late to learn french):

| labels | economie    | sante       | sport    | culture  | sciences      |
|--------|-------------|-------------|----------|----------|---------------|
| 1      | entreprises | direct      | football | film     | pourquoi      |
| 2      | auto        | coronavirus | club     | cinéma   | scientifiques |
| 3      | entreprise  | pourquoi    | tour     | festival | nasa          |
| 4      | salariés    | vaccin      | joueurs  | artistes | étude         |
| 5      | économie    | ehpad       | ligue    | musée    | terre         |
| 6      | secteur     | covid       | psg      | roman    | recherche     |
| 7      | tourisme    | risque      | sport    | art      | science       |
| 8      | marque      | enfants     | équipe   | théâtre  | scientifique  |
| 9      | sncf        | chercheurs  | saison   | culture  | espèces       |
| 10     | marché      | masque      | match    | série    | mars          |

 There are a few misplaced words but globally the result is quite accurate.
 
 Using [sklearn TSNE](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html) we can project the vector space down to two dimensions to be able to draw it:

<img src="https://user-images.githubusercontent.com/44334351/97606534-2e172600-1a10-11eb-9856-22b2ee9d3294.png" width="800">

There is room for improvement in the selection of the 200 out of 80'000 words to display. Some of the groupings around the labels are interesting.
