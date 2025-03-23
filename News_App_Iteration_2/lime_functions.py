# lime_functions.py - functions for outputting LIME explanations and visualizations

import os
# Imports the required libraries for text and data processing
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("stopwords")
# Creates a set (for efficiency) out of English stopwords to use to filter important word features
stop_words = set(stopwords.words('english'))
import spacy
# Emotion lexicon for extracting positive and trust emotions scores for inputted news text
from nrclex import NRCLex
# textstat library for extracting readability features for inputted news text
import textstat
# Visualization libraries and Streamlit
import altair as alt
import streamlit as st
# Imports the LIME explanation library for text explanations
from lime.lime_text import LimeTextExplainer


class BasicFeatureExtractor:
    """
        A class containing methods for extracting the selected discriminative features for 
        helping the machine-learning based classifier categorize real and fake news, including features such as
        normalized (by text length in word tokens) exclamation point count, third person pronoun frequency 
        and PERSON / CARDINAL named entity frequency counts.
    """
    
    def __init__(self):
        # Load in the spacy model
        self.nlp = spacy.load("spacy_model")

    def extractExclamationPointFreqs(self, text):
        """
        Extracts the normalized (by num tokens) frequencies of exclamation points from a news text. 
        To be applied using .apply or .progress_apply to a "text" column in a news DataFrame
        
            Input Parameters:
                text (str): the news text to extract exclamation point frequencies from
    
            Output:
                excl_point_freq (float): the normalized exclamation point frequency for the text
        """
        # Counts the number of exclamation points in the text
        exclamation_count = text.count('!')
        # Tokenizes the text for calculating text length
        word_tokens = word_tokenize(text)
        # Gets the text length in number of word tokens
        text_length = len(word_tokens)
        # Normalizes the exclamation point frequency
        return exclamation_count / text_length if text_length > 0 else 0 # Handle division-by-zero errs
    


    def extractThirdPersonPronounFreqs(self, text):
        """
        Extracts the normalized frequency counts of third-person pronouns in the inputted news text.
        
            Input Parameters:
                text (str): the news text to extract pronoun features from.
            
            Output:
                float: Normalized third-person pronoun frequency.
        """
        # Defines a list of English third-person pronouns
        third_person_pronouns = [
            "he","he'd", "he's", "him", "his",
            "her", "hers", 
            "it", "it's", "its",
            "one", "one's", 
            "their", "theirs", "them","they", "they'd", "they'll", "they're", "they've",   
            "she", "she'd", "she's"
        ]

        # Tokenizes the text for calculating the text length in word tokens
        word_tokens = word_tokenize(text)

        # Calculates the text length in number of word tokens
        text_length = len(word_tokens)

        # Counts the frequency of third-person pronouns in the news text, lower to match the list of third-person pronouns above
        third_person_count = sum(1 for token in word_tokens if token.lower() in third_person_pronouns)

        # Normalizes the frequency by text length in word tokens
        return third_person_count / text_length if text_length > 0 else 0



    def extractNounToVerbRatios(self, text):
        """
        Calculates the ratio of all types of nouns to all types of verbs in the text
        using the Penn Treebank POS Tagset and the SpaCy library with the downloaded
        "en_core_web_lg" model.
        
            Input Parameters:
                text (str): the news text to extract noun-verb ratio features from.
            
            Output:
                float: Noun-to-verb ratio, or 0.0 if no verbs are present.
        """
        
        # Converts the text to an NLP doc object using the SpaCy library.
        doc = self.nlp(text)
        
        # Defines the Penn Treebank POS tag categories for nouns and verbs
        # Reference here: https://www.ling.upenn.edu/courses/Fall_2003/ling001/penn_treebank_pos.html
        noun_tags = ["NN", "NNS", "NNP", "NNPS"]
        verb_tags = ["VB", "VBD", "VBG", "VBN", "VBP", "VBZ"]
        
        # Counts the nouns and verbs based on the Penn Treebank tags
        noun_count = sum(1 for token in doc if token.tag_ in noun_tags)
        verb_count = sum(1 for token in doc if token.tag_ in verb_tags)
        
        # Computes and returns the noun-to-verb ratio (should be higher for fake news, more nouns)
        return noun_count / verb_count if verb_count > 0 else 0.0 # Avoid division-by-zero error


    def extractCARDINALNamedEntityFreqs(self, text):
        """
        Extracts the normalized frequency of CARDINAL named entities in the text
        using the SpaCy library.
        
            Input Parameters:
                text (str): The text to extract the CARDINAL named entity frequencies from.
            
            Output:
                float: Normalized frequency (by number of tokens in the text) of CARDINAL named entities.
        """
        # Processes the text with SpaCy to get NLP doc object
        doc = self.nlp(text)

        # Counts how many named entities have the label "CARDINAL"
        cardinal_entity_count = sum(1 for entity in doc.ents if entity.label_ == "CARDINAL")

        # Tokenizes the text
        word_tokens = [token for token in doc]
        
        # Counts the number of word tokens in the text
        text_length = len(word_tokens)

        # Returns the normalized frequency of CARDIAL named entities
        return cardinal_entity_count / text_length if text_length > 0 else 0.0 # Avoid division-by-zero error


    def extractPERSONNamedEntityFreqs(self, text):
        """
        Extracts the normalized frequency of PERSON named entities in the text
        using the SpaCy library.
        
            Input Parameters:
                text (str): The text to extract the PERSON named entity frequencies from.
            
            Output:
                float: Normalized frequency (by number of tokens in the text) of PERSON named entities.
        """
        # Processes the text with SpaCy to get NLP doc object
        doc = self.nlp(text)
        
        # Counts how many named entities have the label "PERSON"
        person_entity_count = sum(1 for entity in doc.ents if entity.label_ == "PERSON")
        
        # Tokenizes the text
        word_tokens = [token for token in doc]
        
        # Counts the number of word tokens in the text
        text_length = len(word_tokens)
        
        # Returns the normalized frequency of PERSON named entities
        return person_entity_count / text_length if text_length > 0 else 0.0 # Avoids division-by-zero error


    def extractPositiveNRCLexiconEmotionScore(self, text):
        """
        Extracts the POSITIVE emotion score using the NRC Lexicon from the inputted news text.
        
            Input Parameters:
                text (str): the news text to extract POSITIVE emotion score from.
            
            Output:
                float: POSITIVE emotion score.
        """
        # Converts the text to lowercase
        text = text.lower()

        # Creates a NRC Emotion Lexicon object
        emotion_obj = NRCLex(text)
        
        # Returns the POSITIVE emotion score (use "get" to default to 0.0 if for some reason it is not found)
        return emotion_obj.affect_frequencies.get("positive", 0.0) 


    def extractTrustNRCLexiconEmotionScore(self, text):
        """
        Extracts the TRUST emotion score using the NRC Lexicon from the inputted news text.
        
            Input Parameters:
                text (str): the news text to extract TRUST emotion score from.
            
            Output:
                float: TRUST emotion score.
        """
        # Lowercases the text
        text = text.lower()

        # Creates a NRC Emotion Lexicon object
        emotion_obj = NRCLex(text)
        
        # Returns the TRUST emotion score (uses "get" to default to 0.0 if for some reason it is not found)
        return emotion_obj.affect_frequencies.get("trust", 0.0)


    def extractFleschKincaidGradeLevel(self, text):
        """
        Extracts the Flesch-Kincaid Grade Level score for the input text.
        
        Input Parameters:
            text (str): the news text to calculate the Flesch-Kincaid Grade Level score.
        
        Output:
            float: the Flesch-Kincaid Grade Level score for the text.
        """
        return textstat.flesch_kincaid_grade(text)
        
    def extractDifficultWordsScore(self, text):
        """
        Extracts the number of difficult words in the input text using the textstat library.
        
        Input Parameters:
            text (str): the news text to calculate the difficult words score.
        
        Output:
            float: the number of difficult words score for the text.
        """
        # Converts the text to lowercase
        text = text.lower()
        return textstat.difficult_words(text)
    

    def extractCapitalLetterFreqs(self, text):
        """
        Extracts the normalized frequency of capital letters in the input text.
        Normalized by the total number of tokens to account for varying text lengths.
    
            Input Parameters:
                text (str): The news text to extract capital letter frequencies from.
            
            Output:
                float: Normalized frequency of capital letters in the text.
        """
        # Counts the number of capital letters in the text
        capital_count = sum(1 for char in text if char.isupper())
        
        # Tokenizes the text
        word_tokens = word_tokenize(text)

        # Counts number of word tokens in the text
        text_length = len(word_tokens)
        
        # Normalize the frequency of capital letters
        return capital_count / text_length if text_length > 0 else 0.0 # Avoids division-by-zero error
    
    def extractBasicFeatures(self, df, dataset_name, root_save_path="../FPData/BasicFeatureExtractionDFs",):
        """
        Adds new columns to an inputted DataFrame that stores the frequency of 
        or score for various features derived from the "text" column, hopefully for 
        improved classification performance using a Passive Aggressive Classifier model.

            Input Parameters:
                df (pd.DataFrame): DataFrame with a "text" column containing news texts.
    
            Output:
                pd.DataFrame: a new DataFrame with the additional feature columns
        """
        # Creates a copy of the original DataFrame to store new features in
        basic_feature_df = df.copy()

        # Gets the dataset name without the ".csv" at the end to print out dataset name being processed
        dataset_name_without_csv_extension = dataset_name.split('.')[0]
        print(f"Extracting features for {dataset_name_without_csv_extension}...")
        
        # Applies the exclamation point frequency helper function...
        print("Extracting normalized exclamation point frequencies...")
        basic_feature_df["exclamation_point_frequency"] = basic_feature_df["text"].progress_apply(self.extractExclamationPointFreqs)
        
        # Applies the third-person pronoun frequency helper function...
        print("Extracting third person pronoun frequencies...")
        basic_feature_df["third_person_pronoun_frequency"] = basic_feature_df["text"].progress_apply(self.extractThirdPersonPronounFreqs)

        # Applies the noun-to-verb ratio helper functionn
        print("Extracting noun-to-verb ratio frequencies...")
        basic_feature_df["noun_to_verb_ratio"] = basic_feature_df["text"].progress_apply(self.extractNounToVerbRatios)

        # Applies the CARDINAL named entity frequency helper function to create a new column
        print("Extracting CARDINAL named entity frequencies...")
        basic_feature_df["cardinal_named_entity_frequency"] = basic_feature_df["text"].progress_apply(self.extractCARDINALNamedEntityFreqs)

        # Applies the PERSON named entity frequency helper function to create a new column
        print("Extracting PERSON named entity frequencies...")
        basic_feature_df["person_named_entity_frequency"] = basic_feature_df["text"].progress_apply(self.extractPERSONNamedEntityFreqs)

        # Applies the NRC Lexicon to get the POSITIVE emotion scores for each text
        print("Extracting NRC Lexicon POSITIVE emotion scores...")
        basic_feature_df["nrc_positive_emotion_score"] = basic_feature_df["text"].progress_apply(self.extractPositiveNRCLexiconEmotionScore)

        # Applies the NRC Lexicon to get the TRUST emotion scores for each text
        print("Extracting NRC Lexicon TRUST emotion scores...")
        basic_feature_df["nrc_trust_emotion_score"] = basic_feature_df["text"].progress_apply(self.extractTrustNRCLexiconEmotionScore)

        # Apply textstat package to get the Flesch-Kincaid U.S. Grade Level readability score
        print("Extracting Flesch-Kincaid U.S. Grade readability scores...")
        basic_feature_df["flesch_kincaid_readability_score"] = basic_feature_df["text"].progress_apply(self.extractFleschKincaidGradeLevel)

        # Applies textstat package to get the difficult_words readability score
        print("Extracting difficult words readability scores...")
        basic_feature_df["difficult_words_readability_score"] = basic_feature_df["text"].progress_apply(self.extractDifficultWordsScore)

        # Extracts the counts of capital letters for each text normalized by number of tokens
        print("Extracting normalized capital letter frequency scores..")
        basic_feature_df["capital_letter_frequency"] = basic_feature_df["text"].progress_apply(self.extractCapitalLetterFreqs)

        # Saves the dataset with extracted features to disk
        basic_feature_df.to_csv(os.path.join(root_save_path, dataset_name), index=False)
        
        return basic_feature_df

    def extractExtraFeaturesColumns(self, df):
        """
            Extracts the engineered features only from the DataFrame, excluding the other (e.g. text) columns.

            Input Parameters:
                df (pd.DataFrame): news dataset to extract the engineered features from
            Output:
                pd.DataFrame: the extracted engineered feature columns from the DataFrame that was passed in
        """
        return df[[
        "exclamation_point_frequency", "third_person_pronoun_frequency", "noun_to_verb_ratio",
        "cardinal_named_entity_frequency", "person_named_entity_frequency", 
        "nrc_positive_emotion_score", "nrc_trust_emotion_score", 
        "flesch_kincaid_readability_score", "difficult_words_readability_score", 
        "capital_letter_frequency"
    ]] 

    def extractFeaturesForSingleText(self, text):
        """
        Extracts the basic features for classifying the text, to concatenate with the TF-IDF features,
        for a single text instance.
    
        Input Parameters:
            text (str): The text to extract features from.
    
        Output:
            pd.DataFrame: A DataFrame storing the extracted basic feature values for this text, each feature per column
        """
        # Extracts the features and store them in a dict
        feature_dict = {
            "exclamation_point_frequency": self.extractExclamationPointFreqs(text),
            "third_person_pronoun_frequency": self.extractThirdPersonPronounFreqs(text),
            "noun_to_verb_ratio": self.extractNounToVerbRatios(text),
            "cardinal_named_entity_frequency": self.extractCARDINALNamedEntityFreqs(text),
            "person_named_entity_frequency": self.extractPERSONNamedEntityFreqs(text),
            "nrc_positive_emotion_score": self.extractPositiveNRCLexiconEmotionScore(text),
            "nrc_trust_emotion_score": self.extractTrustNRCLexiconEmotionScore(text),
            "flesch_kincaid_readability_score": self.extractFleschKincaidGradeLevel(text),
            "difficult_words_readability_score": self.extractDifficultWordsScore(text),
            "capital_letter_frequency": self.extractCapitalLetterFreqs(text),
        }

        # Converts the dict above to a DataFrame (it will contain a single row, as this is just one text)
        feature_df = pd.DataFrame([feature_dict])

        # Returns the DataFrame
        return feature_df



def explainPredictionWithLIME(
        trained_pipeline,
        text,
        feature_extractor,
        num_features=50,
        num_perturbed_samples=500
    ):
    """
    Extracts the features from text, predicts the inputted news text's probability of being fake news by using a pre-loaded trained TF-IDF, 
    Passive-Agressive Classifier and CalibratedClassifierCV pipeline. Uses LIME to generate local explanations
    for this prediction.

    Input Parameters:
        trained_pipeline (sklearn.pipeline.Pipeline): a pre-trained Pipeline consisting of TF-IDF and a Passive-Aggressive Classifier contained
        within a Calibrated ClassifierCV for returning probabilities.
        text (str): the text to get the real or fake prediction for
        feature_extractor (an instance of BasicFeatureExtractor class): class for extracting extra semantic and linguistic features
                            from the text
        num_features (int): the number of "important features" contributing to the final prediction that the LIME explainer should
                            output
        num_perturbed_samples (int): the number of times LIME should perturb-and-evaluate the entered original text to see how
                                     changing certain features leads to differences in predicted real or fake news probabilities
    Output:
        dict: this stores a summary of the information for explaining the prediction using LIME. It contains:
                - "explanation_object": the default LIME explanation object returned by the LIME text explainer
                - "word_features_list": the list of tuples containing words and their LIME importance scores
                - "extra_features_list": a list of tuples containing the extra engineered features and their importance scores
                - "highlighted_text": the original text string formatted with HTML markup to highlight the important word features
                - "probabilities": an array of probabilities for [real, fake] news
                - "main_predicition": the final prediction, 0 for real news, 1 for fake news
    """

    # Instantiates a Lime Text Explainer
    text_explainer = LimeTextExplainer(class_names=["real", "fake"]) # Map integers to their class names
    
    # Extracts the semantic and linguistic features for the single text using the feature extractor, returning a DataFrame with a single row
    single_text_df = feature_extractor.extractFeaturesForSingleText(text)

    # Adds the original text as a column to the resulting features DataFrame
    single_text_df["text"] = text
    
    # Extracts ONLY the extra features (not the original text) and store in extra_features for generating feature-based LIME explanations
    extra_features = single_text_df.drop("text", axis=1)

    # Extracts the extra feature names from the extra features DataFrame
    extra_feature_names = extra_features.columns  # Extract the names of the extracted features

    def predict_fn(perturbed_texts):
        """
            LIME works "by randomly perturbing features from the instance" / news text.
            Reference: https://lime-ml.readthedocs.io/en/latest/lime.html
            Its .explain_instance methods requires a classifier prediction wrapper function that outputs prediction probabilities
            for the set of randomly perturbed texts.
            This inner (nested) function processes (extracts features using TF-IDF and the Feature Extractor class above) a set of texts
            and then predicts their probabilities (as a 2-element array), using the trained Pipeline to return arrays of 
            probabilities [prob_real, prob_fake] for each perturbed text. Defined inside of the LIME function to have access
            to all the variables such as model etc. and provide encapsulation (as this is only used by the LIME explainer).

            Reference: https://realpython.com/inner-functions-what-are-they-good-for/#:~:text=Inner%20functions%2C%20also%20known%20as,closure%20factories%20and%20decorator%20functions.

            Reference from https://coderzcolumn.com/tutorials/machine-learning/how-to-use-lime-to-understand-sklearn-models-predictions#google_vignette:
                predict_fn: "It accepts prediction function which takes as input sample passed to data_row as input and generates an actual prediction 
                    for regression tasks and class probabilities for classification tasks."

            Input Parameters:  
                perturbed_texts (list): list of the perturbed texts generated by LIME text explainer
            
            Output:
                probs_for_text (numpy.ndarray): an array of probabilities for [real, fake] news classes
        """

        # Stores the DataFrames for the extracted features and the text for all of the randomly perturbed texts
        perturbed_text_features_df_list = []  
        
        # Iterates over perturbed LIME-texts
        for perturbed_text in perturbed_texts:

            # Extracts features for each perturbation of original text, this outputs a DataFrame
            df = feature_extractor.extractFeaturesForSingleText(perturbed_text)

            # Adds the perturbed text to a new column in the extracted features single-row DataFrame
            df["text"] = perturbed_text

            # Adds the single row DataFrame storing the extracted features and perturbed text to the list above
            perturbed_text_features_df_list.append(df)
            
        # Combines the single-row DataFrames for all perturbed LIME texts into a single DataFrame.
        # Stacks the rows using axis=0 argument, remove the index
        combined_df = pd.concat(perturbed_text_features_df_list, axis=0, ignore_index=True)
        
        # Extracts the probabilities of all the perturbed feature-text LIME samples using the pre-trained pipeline
        probs_for_texts = trained_pipeline.predict_proba(combined_df)
        
        # Returns the probabilities of the perturbed texts
        return probs_for_texts

    # Uses the pre-trained model to get the predicted label for a single text (designed for a batch, so get first text prediction at index 0)
    prediction = trained_pipeline.predict(single_text_df)[0]

    # Generates the LIME explanation using .explain_instance()
    # Reference: https://lime-ml.readthedocs.io/en/latest/lime.html
    """"
        Docs: "First, we generate neighborhood data by randomly perturbing features from the instance [...]
        We then learn locally weighted linear models on this neighborhood data to explain each of the classes in an interpretable way (see lime_base.py)."
    """
    explanation = text_explainer.explain_instance(
        text, # The original entered news text
        # Docs: a required "classifier prediction probability function, which takes a numpy array and outputs prediction
        #  probabilities. For ScikitClassifiers, this is classifier.predict_proba.""
        predict_fn, # Nested function for extracting the extra features for each perturbed sample of the text and getting prediciton probs
        num_features=num_features, # Number of top word-features to output in explanation (default = 50)
        top_labels=1, # Generate explanations only for the top-predicted label
        num_samples=num_perturbed_samples, # Number of perturbed versions to generate: a higher value inc. accuracy but takes more time. Default = 500
        labels=[prediction] # Put prediction into a list, has to be an iterable
    )
    
    # Returns the word feature explanations as a list of tuples with scores for the predicted label
    word_features = explanation.as_list(label=prediction)

    # Converts the outputted word features to a list of tuples storing (word, importance_score)
    # Filtesr out stopwords from important word features like "the", "an", etc. to make word features more meaningful to users
    word_features_filtered = [(word_feature[0], word_feature[1]) for word_feature in word_features
                             if word_feature[0].lower() not in stop_words] 
    
    # Sorts the text_feature_list in descending order by absolute value of importance scores
    word_feature_list_sorted = sorted(word_features_filtered, key = lambda x: abs(x[1]), reverse=True)
    
    # Creates a list to store the extra features'importance scores as tuples (feature_name, feature_importance)
    extra_feature_importances= []

    # Extracts the probability array [real_news_probability, fake_news_probability] for the single, unperturbed news text
    original_text_probability_array = trained_pipeline.predict_proba(single_text_df)[0]

    # Iterates through the extra engineered features to compute their importance for the prediction
    for feature in extra_feature_names:

        # Creates a perturbed version of the DataFrame storing the original news text and extra features
        perturbed_df = single_text_df.copy()
        
        # Calculates this particular feature's importance for the prediction by setting its value to 0, and 
        # using the pipeline to generate predicted probabilities WITHOUT this feature by setting its value
        # to 0
        perturbed_df[feature] = 0

        # Passes the sample to the pipeline, including text and extra features (with this feature's value set to zero)
        perturbed_probability_array = trained_pipeline.predict_proba(perturbed_df)[0]
        
        # Calculates the feature's importance as difference in probabilities of the original prediction probability
        importance = original_text_probability_array[prediction] - perturbed_probability_array[prediction]

        # Appends the importance of the feature to the extra_feature_importances list-of-tuples
        extra_feature_importances.append((feature, importance))
    
    # Sorts the extra features by absolute value of the feature importance scores (at index 1 in the name-score tuples), desc. order
    extra_feature_importances.sort(key = lambda x: abs(x[1]), reverse=True)

    # Generates a version of the original text string with highlighted color-coded words in HTML based on their importance scores
    highlighted_text = highlightText(text, word_feature_list_sorted)
    
    # Combine all aspects of the explanation into a dictionary 
    return {
        "explanation_object": explanation,
        "word_features_list": word_feature_list_sorted,
        "extra_features_list": extra_feature_importances,
        "highlighted_text": highlighted_text,
        "probabilities": original_text_probability_array,
        "main_prediction": prediction
    }



def highlightText(text, word_feature_list):
    """
    A helper function for highlighting the words in the input text based on their importance scores and their impact on the prediction.
    Red is used to highlight words pushing towards fake news (label=1) and blue for words pushing towards real news (label=0).

    Input Parameters:
        text (str): the text to highlight
        word_feature_list (list of tuples): list of word-feature, importance-score tuples outputted by the LIME text explainer

    Output:
        str: HTML formatted string with tags designating the highlighted text
    """
    
    # Stores a series of dictionaries containing positions, colors and opacities 
    # for highlighting parts of the text with HTML tags based on word importance
    highlight_positions = []
    
    # Calculates the maximum absolute importance score from all the word features
    max_importance = max(abs(importance) for feature, importance in word_feature_list)

    # Finds all the important word positions to highlight in the original text
    for word_feature, importance in word_feature_list:
        
        # A placeholder for where to start searching for in the original text (to highlight the important words)
        # This will be updated throughout the loop
        pos = 0 # Starts at first character in the text string

        # While-loop runs until all the current word (feature) positions have been found for highlighting
        while True:

            # Finds the first occurrence of the word feature (lowercase for matching) in the original text, starting from "pos"
            pos = text.lower().find(word_feature.lower(), pos)
            
            # If the word is not found, .find() returns -1, so breaks out of the while loop for this particular feature
            if pos == -1:
                break
                
            # Check if the found word is a whole word (not matching solely on just a sub-component of a word, like "portion" in "proportion")
            boundary_positions = detectWordBoundaries(text, pos, word_feature)

            # If the match is just PART of a word, then move to next position in the string after this match 
            # if the found by incrementing pos by 1, and move back to beginning of the while loop to check for
            #  the next occurence of this word in the text
            if boundary_positions is None:
                pos += 1 
                continue
            
            # Unpacks the word beginning and end positions from the returned object if it is not None and this is a whole word
            word_start_pos, word_end_pos = boundary_positions
            
            # Adds the positions for highlighting to the overall dict for the whole text
            highlight_positions.append({
                "start": word_start_pos, # Beginning character position in string of the word to highlight
                "end": word_end_pos, # End character position in string to stop highligthing after
                "color": "red" if importance > 0 else "blue", # Color is based on prediction the important word pushes classifier towards
                "opacity": abs(importance) / max_importance if max_importance != 0 else 0, # Set opacity based on absolute importance strength
                "text": text[word_start_pos:word_end_pos] # Selects the word for highlighting using start and end indices
            })
            
            # Moves past this word by incrementing the string index to be at the character just following this word
            pos = word_end_pos  

    # Sorts the word position dictionaries in ascending order based on word start index
    highlight_positions.sort(key = lambda x: x["start"])
    
    # Merges the adjacent highlights of the same color to represent bigrams and trigrams
    merged_positions = []

    # If there is one or more dictionary in the highlight_positions list then proceed
    if highlight_positions:

        # Extracts the first dictionary (text segment information) to highlight 
        current = highlight_positions[0]

        # Iterates through all of the next highlighting information dictionaries after the current one
        for next_pos in highlight_positions[1:]:

            # Checks if the next highlight segment starts right after current one ends (or overlaps) and has same color
            # (is associated with the same prediction)
            if (next_pos["start"] <= current["end"] + 1 and # Adds 1 to account for spaces between the words
                next_pos["color"] == current["color"]):

                # Merges same-color words by extending current end position to be the one at the end of the next segment instead
                current["end"] = max(current["end"], next_pos["end"])

                # Uses whichever opacity was stronger (less transparency) between the two segments to be merged
                current["opacity"] = max(current["opacity"], next_pos["opacity"])

                # Updates the text to be highlighted based on the news positions
                current["text"] = text[current["start"]:current["end"]]
            else:
                # If is not possible merge with the next important word because they don't overlap, then
                # add the current highlight info to the main list, move to next dictionary by updating current variable
                if next_pos["start"] > current["end"]: 
                    merged_positions.append(current)
                    current = next_pos
                else:
                    # If overlapping words have different colors (blue for real and red for fake), skip the current word if it has the
                    # weaker importance or opacity
                    if next_pos["opacity"] > current["opacity"]:
                        current = next_pos
        
        # Adds the merged highlighted section
        merged_positions.append(current) 

    # Creates a list for building the text highlighted using HTML tags
    result = []

    # Uses this to track rest of text after highlighted sections are finished
    last_end = 0
    
    # Iterates over the dicts specifying highlighting positions and opacities (including merged consecutive sections functionalty)
    for pos in merged_positions:

        # If the next segment to highlight's start position is greater than last_end, append the (non-highlighted) text 
        # before this segemnt to the result list
        if pos["start"] > last_end:
            result.append(text[last_end:pos["start"]])
        
        # Maps the segment dict's information to the appropriate opacity andcolor to use in in-line CSS HTML tags 
        color = "rgba(255, 0, 0," if pos["color"] == "red" else "rgba(100, 149, 255,"  # red or bright blue
        # Gets the value for the alpha opacity channel
        background_color = f"{color}{pos['opacity']})"
        
        # Appends the highlighted text with span HTML tag and inline CSS styling to the result
        result.append(
            f"<span style='background-color:{background_color}; font-weight: bold'>"
            f"{pos['text']}</span>"
        )
        
        # Updates last_end tracker to the end index of the highlighted section
        last_end = pos["end"]
    
    # After done iterating through all highlighted segments, add any remaining text after last_end to the result list
    if last_end < len(text):
        result.append(text[last_end:])

    # Rejoins the result list containing HTML for highlighting into a string
    return "".join(result)



def detectWordBoundaries(text, word_start_pos, word):
    """
    Ensures that only whole words are matched when highlighting text, and not parts of 
    words (e.g. avoid highlighting the word feature "hand" when iterating over the word "handle")

    Returns the start and end position if it's a valid word boundary (start and ende of word), None otherwise

    Input Parameters:
        text (str): the whole text the word to highlight is part of
        word_start_pos (int): starting position of word (feature)
        word (str): the word feature that should be highlighted

    Output:
        tuple of word_start_pos (int), word_end_pos (int): return positions only if word is indeed a word surrounded by a boundary
        None: returns None if there is no valid word boundary around this instance of the word substring
    """

    # Defines punctuation characters for detecting word boundaries, such as space, exclamation mark, hyphen, quotes, newline, tab
    boundary_chars = set(' .,!?;:()[]{}"\n\t-')
    
    # Checks for word boundary at start of the word to highlight: passes check if position is 0 (start of text) or if previous character is in boundary_chars
    start_check = word_start_pos == 0 or text[word_start_pos - 1] in boundary_chars

    # Calculates the end position by adding start idx to length of tword
    word_end_pos = word_start_pos + len(word)

    # Checks for the word boundary at end of word to highlight: it passes if either the word end idx is the last pos in 
    # whole text or next character is in Boundary
    end_check = word_end_pos == len(text) or text[word_end_pos] in boundary_chars

    # If both start_check and end_check are legit, return the word start and end positions
    if start_check and end_check:
        return word_start_pos, word_end_pos

    # If the word is part of a larger word, return None instead of the positions
    return None


def displayAnalysisResults(explanation_dict, container, news_text, feature_extractor, FEATURE_EXPLANATIONS):

    """
    Displays comprehensive LIME prediction analysis results including predivted label, prediction probabilities, 
    confidence scores, and feature importance charts.
    
    Input Parameters:
        explanation_dict (dict): dict containing LIME explanation results, including word and extra feature scores, 
                                and HTML-marked up highlighted text
        container: the instance of the Streamlit app container to display the results in (passed in inside of app.py)
        news_text (str): the user's inputted text to generate prediction and LIME explanations for
        feature_extractor (an instance of the BasicFeatureExtractor class): for processing the inputted text to get 
                                                                            semantic and linguistic features
        FEATURE_EXPLANATIONS (dict): natural language explanations of the different exra engineered non-word semantic and 
                                    linguistic features
    """
    
    # Converts the news category label from 0/1 to text labels
    main_prediction = "Fake News" if explanation_dict["main_prediction"] == 1 else "Real News"
    
    # Extracts the [real, fake] probability array returned from LIME explainer function
    probs = explanation_dict["probabilities"]
    
    # Displays the results based on LIME explainer output
    container.subheader("Text Analysis Results")

    # Displays the general predicted label
    container.write(f"**General Prediction:** {main_prediction}")

    # Writes the probabilities of being real and fake news
    container.write(f"**Confidence Scores:**")
    container.write(f"- Real News: {probs[0]:.2%}")
    container.write(f"- Fake News: {probs[1]:.2%}")
    
    # Displays the highlighted text title using Markdown and inline styling to adjust size and padding
    st.markdown("""
        <div style='padding-top: 20px; padding-bottom:10px; font-size: 24px; font-weight: bold;'>
            Highlighted Text Section
        </div>
        """, unsafe_allow_html=True  # Reference: https://discuss.streamlit.io/t/unsafe-allow-html-in-code-block/54093
    )
    
    # Displays the explanation for the highlighted text
    st.markdown("""
        <div style='padding-bottom:12px; padding-top: 12px; font-size: 18px; font-style: italic;'>
        Highlighted text shows the words (features) pushing the prediction towards <span style="color: blue; font-weight: bold;">real news</span> in blue 
        and to <span style="color: red; font-weight: bold;">fake news</span> in red.
        </div>
    """, unsafe_allow_html=True)
    
    # Reference: https://docs.streamlit.io/develop/api-reference/layout/st.expander
    # Ïnserts a multi-element container that can be expanded/collapsed.""
    with st.expander("View Highlighted Text:"):
        # Format the expandable scroll-box to show highlighted (blue=real, red=fake) text outputted by LIME Explainer using inline CSS to allow y-scrolling and padding
        st.markdown("""
            <div style='height: 450px; overflow-y: scroll; border: 2px solid #d3d3d3; padding: 12px;'>
                {}
            </div>  
            """.format(explanation_dict["highlighted_text"]), unsafe_allow_html=True)
    
    # Title for bar charts for feature importance analysis
    container.subheader("Feature Importance Analysis")

    # Creates two columns for side-by-side bar charts for word features on the left and extra features on the right
    col1, col2 = container.columns(2)
    
    # First column: word features
    with col1:
        col1.write("### Top Word Features")

        # Ges the top text-based features identified by LIME into a DataFrame for easier sorting and filtering
        word_features_df = pd.DataFrame(
            explanation_dict["word_features_list"],
            columns=["Feature", "Importance"]
        )

        # Returns the first 10 rows with the largest values in the Importance column, in descending order. 
        # Reference: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.nlargest.html
        word_features_df = word_features_df.nlargest(10, "Importance")
        
        # Creates a bar chart for most important text features using the Altair visualization library using text_features_df
        # Q = specifies this feature/value is quantitative, N = specifies it is nominal/categorical
        # sort="-x" = sort features by x-value (importance score)
        # Reference: https://altair-viz.github.io/user_guide/generated/toplevel/altair.Chart.html#altair.Chart
        # How to create charts in Streamlit Reference: https://www.projectpro.io/recipes/display-charts-altair-library-streamlit
        word_features_chart = alt.Chart(word_features_df).mark_bar().encode( # Use mark_bar to create bar_chart. Reference: https://altair-viz.github.io/user_guide/marks/index.html
            x=alt.X("Importance:Q", title="Importance Strength"), # Q = numerical quantity
            y=alt.Y("Feature:N", sort="-x", title=None, # N = named/categorical value
                    axis=alt.Axis( # For formatting the feature labels, Reference: https://altair-viz.github.io/user_guide/generated/core/altair.Axis.html
                        labelFontSize=10,
                    )), # Customizes the y-axis appearance for label length threshold, size, removing tick-marks etc.
            # Reference for how to determine color in Altair library: https://altair-viz.github.io/user_guide/generated/api/altair.condition.html
            color=alt.condition(
                alt.datum.Importance > 0, # Show features pushing towards fake news (pos importance score) in red, real news in blue
                alt.value("red"), # Positive importance (pushing towards fake)
                alt.value("blue") # Negative importance (-> real)
            ),
            tooltip=["Feature", "Importance"] # Add s"tooltips": explanations that appear when hovering above the bar
        ).properties(
            title="Top 10 Word Features",
            height=400,
            width=500
        ) # Sets the chart's title and dimensions
        
        # Displays the word features chart
        col1.altair_chart(word_features_chart , use_container_width=True)

    # Second column showing the bar chart with importance scores for non-word features
    with col2:

        col2.write("### Top Extra Semantic and Linguistic Features")
        
        # Creates a DataFrame from the extra features list returned by the LIME explainer function
        extra_features_df = pd.DataFrame(
            explanation_dict["extra_features_list"],
            columns=["Feature", "Importance"]
        )
        
        # Sorts the features by their absolute importance
        extra_features_df["Absolute Importance"] = extra_features_df["Importance"].abs() # Create new absolute importance column first
        extra_features_df = extra_features_df.nlargest(10, "Absolute Importance")
        
        # Adds the original feature name before mapping to the explanations
        extra_features_df["Original Feature"] = extra_features_df["Feature"]
        
        # Maps the feature column variables to user-readable strings 
        feature_name_mapping = {
            "exclamation_point_frequency": "Exclamation Point Usage",
            "third_person_pronoun_frequency": "3rd Person Pronoun Usage",
            "noun_to_verb_ratio": "Noun/Verb Ratio",
            "cardinal_named_entity_frequency": "Number Usage",
            "person_named_entity_frequency": "Person Name Usage",
            "nrc_positive_emotion_score": "Sentiment/Emotion Polarity",
            "nrc_trust_emotion_score": "Trust Score",
            "flesch_kincaid_readability_score": "Readability Grade",
            "difficult_words_readability_score": "Complex Words",
            "capital_letter_frequency": "Capital Letter Usage"
        }
        
        # Maps the features to their more readable names listed in the dictionary above
        extra_features_df["Feature"] = extra_features_df["Feature"].map(feature_name_mapping)
        
        # Maps the explanations to their natural language explanation for users
        extra_features_df["Explanation"] = extra_features_df["Original Feature"].map(FEATURE_EXPLANATIONS)
        
        # Creates the bar chart with enhanced tooltips for explaining what features mean to users when they hover over a bar
        extra_features_chart = alt.Chart(extra_features_df).mark_bar().encode(
            x=alt.X("Importance:Q", title="Impact Strength"),
            y=alt.Y("Feature:N",
                    sort=alt.EncodingSortField(
                        field="Absolute Importance", # Sorts the features by ABSOLUTE value of importance
                        order="descending"
                    ),
                    axis=alt.Axis(
                        labelFontSize=10 # Formats the feature label size
                    )), 
            color=alt.condition(
                alt.datum.Importance > 0, # Predicate for the color condition
                alt.value("red"),   # If condition is true (over 0, so show a red bar for fake news)
                alt.value("blue") # If condition is false, and importance is negative show a blue bar for rela news
            ),
            tooltip=["Feature", "Importance", "Explanation"]  # Add tooltip showing feature and importance if user hovers over the bar 
        ).properties(
            title="Top 10 Extra Features",
            height=400,
            width=500
        ) # Adds the title and size
        
        # Shows the extra features chart in the column
        col2.altair_chart(extra_features_chart, use_container_width=True)
        
        # Adds a legend explanation for the color-coded bar charts and highlighted text
        container.markdown(f"""
            **Legend:**
            - 🔵 **Blue bars**: Features pushing towards real news classification
            - 🔴 **Red bars**: Features pushing towards fake news classification
            
            The length of each bar and color represent how strongly this feature
            in the news text pushes the classifier towards a REAL or FAKE prediction.
            For more details about the raw, scaled scores of these features and
            explanations of their distributions in real vs fake training data,
            please click below or on the Key Pattern Visualizations tab above.
        """)
        
                
        # Extracts the actual feature values for the single text news prediction
        single_text_df = feature_extractor.extractFeaturesForSingleText(news_text)
        
        with col2.expander("*View More Detailed Feature Score Information*"):
            
            # Iterates over the extra features to explain each one
            for index, row in extra_features_df.iterrows():
                # Gets the raw-score/actual feature value for the news text
                actual_value = single_text_df[row["Original Feature"]].iloc[0]
                # Determines its color based on its importance value: red if pushing towards positive/fake news, else blue
                importance_color = "red" if row["Importance"] > 0 else "blue"
                # Maps the importance score to string description
                if row["Importance"] > 0:
                    importance_explanation = "(pushing towards fake news)"
                elif row["Importance"] < 0:
                    importance_explanation = "(pushing towards real news)"
                else:
                    importance_explanation = "(neutral/has no impact)"
                
                # Adds some text explaining what exactly this engineered feature and its score means for the prediction
                container.markdown(f"""
                    **{row["Feature"]}**
                    - Raw Score (not importance score, but the actual feature): {actual_value:.4f}
                    - Impact on Classification: <span style='color:{importance_color}'>{row["Importance"]:.4f} {importance_explanation}</span>
                    - {FEATURE_EXPLANATIONS[row["Original Feature"]]}
                    ---
                """, unsafe_allow_html=True)
                
