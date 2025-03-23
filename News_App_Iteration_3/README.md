# Third Iteration of Fake News Detection App

Contains the following new components:

1. More modular code, broken into three main files: app.py (major Streamlit starting point), lime_functions.py (all LIME-related chart visualization and explainability functions),
feature_extractor.py (a class for extracting engineered semantic and linguistic features from the news text)
2. A model pipeline which consists of a supervised FastText model (downloaded from Google Drive) trained on a combined training dataset consisting of WELFake, Constraint, PolitiFact and GossipCop datasets, to try to address the problem of poor cross-domain performance, a pre-fitted scikit-learn StandardScaler fitted on the combined training data, and the final trained Passive-Aggressive Classifier
3. Separated word features pushing towards the main prediction and against it into two separate Altair bar charts in response to user feedback
4. Color coded the probability results with real news in blue and fake news in red for clearer readability
5. More detailed guidance for how the app works


