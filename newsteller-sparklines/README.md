# NewsTeller topic activity sparklines studies

## Table of contents
* [General info](#general-info)
* [Extract topics](#extract-topics)
* [Reorganize data around topics](#reorganize-data-around-topics)
* [Sort topics](#sort-topics)
* [Display sparklines](#display-sparklines)
* [Remarks and leads for future exploration](#remarks-and-leads-for-future-exploration)


## General info
NewsTeller is a research-driven platform to analyze news. This notebook explores topic activity sparklines. The high-level goal was to find "meaningful" sparklines and end up being extracting breaktrough topics. All experiences were made with a subset of one month (10.20.2020 - 11.20.2020)of articles collected by the newsteller platform. The best results were obtained using:
* [SpaCy](https://spacy.io) to extracted nouns out of articles text and titles
* [CountVectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html) to count the number of citations per hour
* Highest peak divided by mean score to sort topics

Giving a lot of sparklines with this kind of shape:
<img width="898" src="https://user-images.githubusercontent.com/44334351/104808703-5a109f00-57e8-11eb-91cf-d7373a651d48.png">

I can easily imagine how to add features to the app / website based on this sparklines exploration (see [Remarks and leads for future exploration](#remarks-and-leads-for-future-exploration)). It's also a open door for future deeper data analysis based on articles collected by the NewsTeller platform.

## Extract topics
In order to find breaktrough topics, we must first extract topics from articles.

### Most frequent words
The first attempt was to count the number of citations for every word. It worked quite well but was noisy with for example "said" in first position.

### TF-IDF
The second attempt was computing TF-IDF score for every word. Surprisingly, the same noisy words were on top.

### Most frequent nouns - spacy
The third and most successful attempt was using [spaCy](https://spacy.io). Spacy is a natural language processing library that extracts nouns out of text, thus reducing our previous noise. (This operation was extremely time consuming on my PC.)

## Reorganize data around topics
Now that we have our topics, we need to transform our panda dataframe regrouping all our articles in something more practical. Using [CountVectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html) I created a date x handle (newspaper) x topic matrix that stores for each topic the number of citations. (I ended up not using the handle dimension at all).

## Sort topics
We want to find meaningful / sensational sparklines. My intuition was that abrupt changes in the sparklines would lead to extraordinary activity about the concerned topic.

### Steepest peak
The first attempt was to sort the topics by the slope of their steepest peak. To be able to do that, I computed the differences matrix by subtracting the one-hour-time-shifted matrix to the original one, then I sorted the topics by comparing their maximum positive slope. The top topics were basically the ones that appeared the most in the articles. (My test time period covers the US election, useless to say that "trump", "biden" and "election" were in top 3). This ordering was not taking into account the extraordinary component, indeed topics with a lot of peaks still managed to have very steep peaks.

### HPM score (Highest peak divided by mean)
The second ordering score now takes into account that the peak must be unusual. By dividing the highest peak by the mean, we select topics with a lot of citations at a given time but that are not constantly in the headlines. It makes sparkline like the one in [General info](#general-info) emerge. The resulting ordering is quite robust, meaning that as far as I explored (top 200) there are very little noise (ie. misplaced topics). This score also allows to specify the period over which we are looking for a peak but still taking all history of the topic into account. 

## Display sparklines
I used [matplotlib](https://matplotlib.org/3.1.1/index.html) to display sparklines. I choosed to print the titles of three related articles in order to give some context to topics.

## Remarks and leads for future exploration
* All required computation for HPM score can be done continously at low cost, it seems imaginable to me to have it eventually deployed in the NewsTeller pipeline. Furthermore, HPM score specifies on what period we are looking for a peak. The way I imagine it to be integrated in the app, would let the user choose between the last day / week / month and the page would display the top topics sparkline in that period. A simple click on the sparkline would redirect the user to the search page with the results of a search for that topic. 
* For the HPM score, I set 1 as lower bound for the mean but in fact most of topic citations mean is below that. I think it could be interesting to rethink the implementation of this score to take this into account.
