# Imports the required libraries
# For Streamlit app
import streamlit as st
# For extracting news articles from URLs
from newspaper import Article
# For loading in graphs and charts showing global fake vs real news patterns
import matplotlib.pyplot as plt
# For model loading
import joblib

# Imports the custom functions from lime_functions.py for generating LIME explanations
from lime_functions import BasicFeatureExtractor, explainPredictionWithLIME, displayAnalysisResults

# Stores a map of feature column names to user-directed detailed explanations for explaining what featues mean
FEATURE_EXPLANATIONS = {
    "exclamation_point_frequency": "Normalized exclamation marks frequency counts. Higher raw scores may indicate more emotional or sensational writing, more associated with fake news in the training data.",
    "third_person_pronoun_frequency": "Normalized frequency of third-person pronouns (he, she, they, etc.). Higher raw scores may indicate narrative and story-telling style. More positive scores associated with fake news than real news in training data.",
    "noun_to_verb_ratio": "Ratio of nouns to verbs. Higher values suggest more descriptive rather than action-focused writing. Higher scores (more nouns to verbs) associated more with real news than fake news in training data. Negative values more associated with fake news than real news.",
    "cardinal_named_entity_frequency": "Normalized frequency of numbers and quantities. Higher scores indicate higher level of specific details, more associated with real news.",
    "person_named_entity_frequency": "Normalized frequency of PERSON named entities. Shows how person-focused the text is, higher scores more associated with fake news.",
    "nrc_positive_emotion_score": "Measure of the positive emotional content using the NRC lexicon. Higher values indicate more positive tone, and more positive tone is associated more with real news than fake news.",
    "nrc_trust_emotion_score": "Measure of trust-related words using NRC lexicon. Higher values suggest more credibility-focused language, and is more associated with real news than fake news.",
    "flesch_kincaid_readability_score": "U.S. grade level required to understand the text. Higher scores indicate more complex writing, which is associated more with real news in the training data.",
    "difficult_words_readability_score": "Count of complex words not included in the Dall-Chall word list. Higher values mean more complex language/vocab is used, which was associated more with real news in the training data.",
    "capital_letter_frequency": "Normalized frequency of capital letters. Higher values might indicate more emphasis or abbreviations to organizations/institutions. Associated more with real news in training data"
}


# Loads model
@st.cache_resource # Saves it for quicker loading next time
def load_pipeline():
    return joblib.load("iteration2_lime_model.pkl")

# Creates instance of the extra engineered feature extractor
feature_extractor = BasicFeatureExtractor()

# Loads the pre-trained TF-IDF and Passive-Aggressive Classifier pipeline
with st.spinner("Loading fake news detection model..."):
    pipeline = load_pipeline()

# Sets an app title
st.title("Fake News Detection App")

# Creates tabs for news text classification and visualizing key patterns
tabs = st.tabs(["Enter News as URL", "Paste in Text Directly", "Key Pattern Visualizations",
                "Word Clouds: Real vs Fake", "How it Works..."])


# First tab: user inputs URL to extract news from
with tabs[0]:

    # Sets the tab heading
    st.header("Paste URL to News Text Here")

    # Creates the text area for entering news URL
    url = st.text_area("Enter news URL for classification", placeholder="Paste your URL here...", height=68)
    
    # Adds slider to allow users select the number of perturbed samples for LIME explanations
    num_perturbed_samples = st.slider(
        "Select the number of perturbed samples for explanation",
        min_value=25,
        max_value=500,
        value=50,  # Default is 50 perturbations
        step=25, # Step size of 25
        help="Increasing this value will make the outputted explanations more accurate but may take longer to process!"
    )
    
    # Adds an accessible explanation for slider values
    st.write("The more perturbed samples you choose, the more accurate the explanations will be, but they will take longer to compute.")
    
    # Creates an interactive button to the classify text and descriptor for this specific tab (that button is for 
    # classifying news URLs not copied and pasted text)
    if st.button("Classify", key="classify_button_url"):
        # Checks if the news URL input is not empty before proceeding
        if url.strip():  
            # Tries to extract the news text from the URL using the newspaper3k library
            try:
                with st.spinner("Extracting news text from URL..."):

                    # Uses the newspaper3k to scrape the article
                    # Reference: https://newspaper.readthedocs.io/en/latest/
                    news_article = Article(url)
                    news_article.download()
                    news_article.parse()
                    news_text = news_article.text
                    
                    # Shows the original text in an expander
                    with st.expander("View the Original News Text"):
                        st.text_area("Original News Text", news_text, height=300)
                    
                    # Generates the prediction and LIME explanation using the custom func
                    with st.spinner("Analyzing text..."):
                        explanation_dict = explainPredictionWithLIME(
                            pipeline,
                            news_text,
                            feature_extractor,
                            num_perturbed_samples=num_perturbed_samples
                        )
                        
                        # Displays the prediction and visualizations
                        displayAnalysisResults(explanation_dict, st, news_text, feature_extractor, FEATURE_EXPLANATIONS)

            # Displays error message if something goes wrong with the explanation generation         
            except Exception as e:
                st.error(f"Error extracting the news text: {e}. Please try a different text!")

        # Displays error if the URL is empty or invalid
        else:
            st.warning("Warning: Please enter some valid news text for classification!")


# Second user option: allows users to insert news text input copied & pasted or written in directly as text
with tabs[1]:
    st.header("Copy and Paste or Write News Text In Here Directly")
    news_text = st.text_area("Paste in the news text for classification",
                             placeholder="Paste your news in here...",
                             height=300)
    
    # Adds the slider to let users control the number of perturbed samples for LIME explanations
    num_perturbed_samples = st.slider(
        "Select the number of perturbed samples to use for the explanation",
        min_value=25,
        max_value=500,
        value=50, 
        step=25,
        help="Warning: Increasing this value will make the outputted explanations more accurate but may take longer to process!"
    )
    
    st.write("The more perturbed samples you choose, the more accurate the explanation will be, but it will take longer to compute.")
    
    if st.button("Classify", key="classify_button_text"):
        # Check if news text is not empty
        if news_text.strip():
            try:
                # Uses the entered text as news text directly for classification
                with st.spinner(f"Analyzing text with {num_perturbed_samples} perturbed samples..."):
                    explanation_dict = explainPredictionWithLIME(
                        pipeline,
                        news_text,
                        feature_extractor,
                        num_perturbed_samples=num_perturbed_samples
                    )
                    
                    # Displays the prediction and visualizations
                    displayAnalysisResults(explanation_dict, st, news_text, feature_extractor, FEATURE_EXPLANATIONS)
                   
            # Displays error if something goes wrong with the explanation      
            except Exception as e:
                st.error(f"Error analyzing the text: {e}")
        # Displays error if the news text was invalid
        else:
            st.warning("Warning: Please enter some valid news text for classification!")

# Third tab: Contains visualizations of REAL vs FAKE news patterns
with tabs[2]:

    # Adds a title and subtitle
    st.header("Bar Charts showing Key Feature Patterns in the Training Data: Real (Blue) vs Fake (Red) News")
    st.markdown("These visualizations show the global patterns distinguishing between real and fake news articles in the training data.")
    
    # Capital Letter frequencies bar chart
    st.subheader("Capital Letter Counts")
    caps_img = plt.imread("all_four_datasets_capitals_bar_chart_real_vs_fake.png")
    st.image(caps_img, caption="Mean number of capital letters in real vs fake news", use_container_width=True)
    st.markdown("Real news tended to use more capital letters, perhaps due to including more proper nouns and technical acronyms.")
    
    # Third Person Pronoun frequencies bar chart
    st.subheader("Third Person Counts")
    pronouns_img = plt.imread("all_four_datasets_third_person_pronouns_bar_chart_real_vs_fake.png")
    st.image(pronouns_img, caption="Frequency of third-person pronouns in real vs fake news", use_container_width=True)
    st.markdown("Fake news often uses more third-person pronouns (e.g him, his, her), which could indciate a more 'storytelling' kind of narrative style.")

    # Exclamation Point frequencies bar chart
    st.subheader("Exclamation Point Counts")
    exclaim_img = plt.imread("all_four_datasets_exclamation_points_bar_chart_real_vs_fake.png")
    st.image(exclaim_img, caption="Frequency of exclamation points in real vs fake news", use_container_width=True)
    st.markdown("Fake news tends to use more exclamation points, possibly suggesting a more sensational and inflammatory writing.")
    
    # Emotion counts (NRC Lexicon) bar chart
    st.subheader("Ten Main Emotion Scores using NRC Emotion Lexicon")
    emotions_img = plt.imread("all_four_datasets_emotions_bar_chart_real_vs_fake.png")
    st.image(emotions_img, caption="Emotional content comparison between real and fake news", use_container_width=True)
    st.markdown("Fake news (in this dataset) often showed lower positive emotion scores and fewer trust-based emotion words than real news.")

    # Section with more detailed explanations about the feature charts
    with st.expander("📊 More details about these features and bar charts"):
        st.markdown("""
        These charts were created using four well-known, benchmark fake news datasets used for training the model:
        WELFake, Constraint (COVID-19 data), PolitiFact (political news), and GossipCop (entertainment + celebrity news). 
                 
        The bar charts show NORMALIZED versions of frequency counts, e.g. for exclamation marks and capital use: this means that the 
        raw counts were divided by the text length (in word tokens) to account for the differences in the lengths 
        of the different news texts, as these varied significantly across the dataset!
        
        **Some of the Main Differences in Real vs Fake News Based on the Data Analysis:**
        
        - Capital Letters: There were higher frequencies in real news, perhaps to greater usage of proper nouns and techical acronyms
        - Third-person Pronouns: These were much more frequent in fake news based on these datasets, suggesting storytelling-like narrative style and person-focused content
        - Exclamation Points: Again, much more frequent in fake news, pointing towards a sensational inflammatory style
        - Emotion Scores for Positive Words and Trust-Related words: The words used in fake news tended to significantly
                     less positive emotional connotations and reduced trust scores
        
        **Disclaimer:** These patterns were extracted from THESE four specific datasets. This information should be considered in combination with other features
        (i.e. the keyword importance scores), as well as remembering that these trends might not generalize well to recent
        news, given the rapid evolution and ever-shifting landscape of fake news and disinformation!
        
        More feature analysis coming soon!
        """)

# Word Clouds visualizing named entity patterns for real vs fake news
with tabs[3]:

    # Adds title + subheader
    st.header("Most Common Named Entities in Real vs Fake News")
    st.markdown("The WordClouds visualize most frequent named entities (e.g. people, countries, companies) in real and"
    " fake news articles from our training data. The size of each word is proportional to how frequently it appears.")
    
    # First WordCloud: named entities which appear only in real news
    st.subheader("Named Entities Appearing ONLY in Real News and NOT in Fake News")
    real_cloud_img = plt.imread("combined_four_set_training_data_real_news_named_entities_wordcloud.png")
    st.image(real_cloud_img, caption="Most frequent entities exclusive to real news data", use_container_width=True)
    
    # Adds more space between WordClouds
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Second WordCloud: named entities appearing only in fake news)
    st.subheader("Named Entities Appearing ONLY in Fake News and NOT in Real News")
    fake_cloud_img = plt.imread("combined_four_set_training_data_fake_news_named_entities_wordcloud.png")
    st.image(fake_cloud_img, caption="Most frequent entities exclusive to fake news data", use_container_width=True)
   
    # Adds even more space between the WordClouds
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Adds explanations for word clouds and how to read them
    with st.expander("☁️ Word Cloud Explanation"):
        st.markdown("""
        The word's size reflects how frequently it occurred in the data for real vs fake news.
        Colors are only used for improving the readability and aesthetics, and do not carry additional meaning.
        """)

# Tab containing explanations for how the LIME algorithm works, for both technical and non-technical users
with tabs[4]:
    st.header("❓ How Does This Work?")
    
    st.write("""
    An algorithm called LIME (Local Interpretable Model-agnostic Explanations) is used in this app to explain the
    individual predictions made for a news item (i.e. whether the news text is real or fake).
             
    Let's get a glimpse into the general intuition behind how this technique works.
             
    """)
    
    st.subheader("⚙️ The Main Idea Behind LIME")
    st.markdown("""
    Whenever this app analyzes a news text, it doesn't just tell you if the news is "fake news" or "real news". The core purpose of
    LIME is to explain which features of the text led the model to make the outputted decision.
    As such, highlights WHICH word features, or more high-level semantic and linguistic features (such as use of certain punctuation marks)
    , in the news text led to the outputted classification. Furthermore, the algorithm also outputs the probability of news being fake,
    rather than a simple label, so that you can get an insight into the certainty of the classifier.
    """)
    
    st.subheader("🍋‍🟩 How Does LIME Generate the Explanations?")
    st.markdown("""
    LIME removes certain words or linguistic features in the news text one-by-one, and runs the trained machine-learning model to see
    how the outputted probabilities change when the text has been slightly changed.

    (a) LIME randomly removes some words from the news input
    (b) It then runs the altered versions of the news texts through the pre-trained classifier and records how much changing these individual features
    has impacted the final prediction
    (c) If changing a specific word has a big impact on the final predicted probability, this feature is then assigned a higher importance
    scor. The importance scores are visualized using bar charts and highlighted text, with red signifying the feature is associated with fake news
    and blue signalling a stronger association with real news.
    """)
    
    st.subheader("📈 Which extra features (apart from words) have been included for making predictions?")
    st.markdown("""
    This model classifies news articles based on the specific features that were found to be the most useful for discriminating 
    between real and fake news based on an comprehensive exploratory data analysis to find the most consistent feature patterns:

    - Individual words appearing more frequently in fake than real news
    - Use of punctuation such as exclamation mark  and capital letters
    - Syntactic patterns such as noun-to-verb ratio
    - The frequency of PERSON and CARDINAL (number) named entities
    - Trust and positive emotion scores using the NRC Lexicon
    - Text readability (how complex the text is to read, e.g. how many difficult words are used based on a list from the "textstat" Python library)
    """)
    
    with st.expander("⁉️ Why Were THESE Particular Features Chosen?"):
        st.markdown("""
        These features were engineered based on a detailed exploratory analysis focusing on the key differences between real and fake news
        over four benchmark datasets: WELFake (general news), Constraint (COVID-19 related health news), PolitiFact (political news),
        and GossipCop (celebrity and entertainment news).
        
        - Fake news is often associated with a more sensational style (e.g. using more exclamation points) than real news, and more "clickbaity" language
        - Real news tends to use more nouns than verbs, as well as more references to numbers, signalling a more factal style
        - Narrative style (e.g. using more third-person pronouns for a more "storytelling" style) can be an indicator of fake news
        - How easy the text is to read and difficult word usage can also help the classifier distinguish between real and fake news,
        as fake news is often easier to digest and less challenging.
        """)
        
    st.subheader("😐 Disclaimer: Limitations of the Model")
    st.markdown("""
        Please remember that strategies for producing fake news are always evolving rapidly, particularly with the rise of generative AI.
        The patterns highlighted here are based on THIS specific training data from four well-known fake news datasets; however,
        they may not apply to newer forms of disinformation!  As a result, it is always strongly recommended to
        also use fact-checking and claim-busting websites to check out whether the sources of information are legitimate.
    """)
    
