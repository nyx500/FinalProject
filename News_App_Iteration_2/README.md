# Second Iteration of Fake News Detection App

Contains the following new components:

1. Tabs containing both a URL input box and text input box for users to either directly paste in a news URL or the text itself
2. A new classifier trained on both WELFake text samples (TF-IDF vectorized) as  well as extra semantic and linguistic engineered features
3. Two charts showing important feature scores, constructed with the Altair library; one for word features + another for the extra engineered features, generated
by creating a new additional feature importance score calculating algorithm
4. An expander showing the highlighted news text with red terms signalling the word is pushing towards fake news, and blue towards real news
5. Bar charts and Word Clouds showing global patterns for real versus fake news in the training data
6. Included a slider for users to determine how many LIME perturbed samples to use, balancing efficiency and the quality of explanations
7. Updated the explanation of how the system works to be more accessible to non-technical users

