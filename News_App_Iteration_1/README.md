# First Iteration of Fake News Detection App

Contains the following components:

1. An input text box where users can copy and paste in news texts
2. A pre-trained TF-IDF and Passive-Aggressive Classifier pipeline trained on the WELFake dataset
3. The LIME TextExplainer outputs a list of "most important words" and their importance scores for the classifier's final decision
4. Users can select an option from a dropdown to either see the main class prediction and importance scores as a text explanation, or as 
a bar chart showing LIME generated important word features pushing towards fake news in red and towards real news in blue
