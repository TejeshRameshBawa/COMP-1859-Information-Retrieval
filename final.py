# Importing Necessary Libraries
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import json


# Function used to get the users choice between EHR or Research Articles
def get_user_choice():
    print("Choose an option:")
    print("1. Search the EHR of Patients")
    print("2. Search Different Types of Research Articles")

    # Loop to ensure valid user input
    while True:
        try:
            # Prompt to get user to enter their choice
            choice = int(input("Enter the number of your choice (1 or 2): "))
            # Validating the user input
            if choice in [1, 2]:
                return choice
            else:
                print("Invalid choice. Please enter 1 or 2.")
        except ValueError:
            # Response invalid input
            print("Invalid input. Please enter a number.")


# Defining the function to get user choice
user_choice = get_user_choice()

# Check user choice and execute actions
if user_choice == 1:
    # Provide feedback to user by stating their choice
    print("You Chose Search the EHR of Patients")

    # Downloading NLTK resources
    nltk.download('punkt')
    nltk.download('stopwords')

    # Tokenization and Preprocessing
    def preprocess_text(text):
        # Tokenize the text
        tokens = nltk.word_tokenize(text)

        # Remove stopwords
        stop_words = set(stopwords.words('english'))
        tokens = [word for word in tokens if word.lower() not in stop_words]

        # Apply stemming (reducing words to their root form)
        stemmer = PorterStemmer()
        tokens = [stemmer.stem(word) for word in tokens]

        return tokens


    # Define the path to your CSV file
    ehr_csv_file_path = "ehr.csv"
    clinical_notes_csv_file_path = "clinical_notes.csv"
    images_path = "images.csv"

    # Read the CSV file using pandas
    ehr_data = pd.read_csv(ehr_csv_file_path, encoding='ISO-8859-1')
    clinical_notes_data = pd.read_csv(clinical_notes_csv_file_path, encoding='ISO-8859-1')
    images_data = pd.read_csv(images_path, encoding='ISO-8859-1')

    # Prompt the user to enter a name and date of birth
    user_name = input("Enter the name: ")
    user_dob = input("Enter the date of birth (DD/MM/YYYY): ")

    # Preprocess the user input
    tokens_user_dob = preprocess_text(user_dob)

    # Retrieve documents for the user's query
    ehr_matching_documents = []

    # Iterate through each record in the CSV
    for index, row in ehr_data.iterrows():
        name = row['Name']
        dob = row['DateOfBirth']

        # Preprocess the content
        tokens_dob = preprocess_text(dob)

        # Check if the user's input matches the record
        if user_name.lower() == name.lower() and tokens_user_dob == tokens_dob:
            ehr_matching_documents.append(row)

    # Initializing an empty list to store the matching documents
    clinical_notes_matching_documents = []

    # Iterating through rows of clinical_notes_data DataFrame
    for index, row in clinical_notes_data.iterrows():
        # Extract relevant information from the rows
        name = row['Name']
        clinical_notes = row['ClinicalNotes']

        # Preprocess clinical notes text
        tokens_clinical_notes = preprocess_text(clinical_notes)

        # Check if the names are matching
        if user_name.lower() == name.lower():
            # Add row
            clinical_notes_matching_documents.append(row)

    # Initializing an empty list to store the matching documents
    images_matching_documents = []

    # Iterating through rows of clinical_notes_data DataFrame
    for index, row in images_data.iterrows():
        # Extract relevant information from the rows
        name = row['Name']
        images = row['Images']

        # Preprocess clinical notes text
        tokens_clinical_notes = preprocess_text(images)

        # Check if the names are matching
        if user_name.lower() == name.lower():
            # Add row
            images_matching_documents.append(row)

    # Display matching documents for EHR
    if ehr_matching_documents:
        print("Matching documents from EHR:")
        for document in ehr_matching_documents:
            print(f"Name: {document['Name']}")
            print(f"Date of Birth: {document['DateOfBirth']}")
            print(f"Gender: {document['Gender']}")
            print(f"Symptoms: {document['Symptoms']}")
            print(f"Causes: {document['Causes']}")
            print(f"Medicine: {document['Medicine']}")
    # feedback when no matching documents are found
    else:
        print("No matching documents found for the given name and date of birth.")

    # Display matching documents for clinical documents
    if clinical_notes_matching_documents:
        print("Matching documents from Clinical Notes:")
        for document in clinical_notes_matching_documents:
            print(f"Clinical Notes: {document['ClinicalNotes']}")
            print(f"Annotation: {document['Annotation']}")
    # feedback when no matching documents are found
    else:
        print("No matching documents found for the given name and date of birth.")

    # Display matching documents for images
    if images_matching_documents:
        print("Matching documents from Images:")
        for document in images_matching_documents:
            print(f"Images: {document['Images']}")
    # feedback when no matching documents are found
    else:
        print("No matching documents found for the given name and date of birth.")


# Check user choice and execute actions
elif user_choice == 2:
    print("You Chose Search Different Types of Research Articles")

    # Downloading NLTK resources
    nltk.download('punkt')
    nltk.download('stopwords')

    # Tokenization and Preprocessing
    def preprocess_text(text):
        # Tokenize the text
        tokens = nltk.word_tokenize(text)

        # Remove stopwords
        stop_words = set(stopwords.words('english'))
        tokens = [word for word in tokens if word.lower() not in stop_words]

        # Apply stemming (reducing words to their root form)
        stemmer = PorterStemmer()
        tokens = [stemmer.stem(word) for word in tokens]

        return tokens


    # Initialize the inverted index
    inverted_index = {}

    # Define the path to your CSV file
    csv_file_path = "research_articles.csv"

    # Read the CSV file using pandas
    research_data = pd.read_csv(csv_file_path, encoding='ISO-8859-1')

    # Iterate through each article in the CSV
    for index, row in research_data.iterrows():
        title = row['Title']
        content = row['Content']

        # Preprocess the content
        tokens = preprocess_text(content)

        # Build the inverted index
        for token in tokens:
            if token in inverted_index:
                inverted_index[token].append(title)
            else:
                inverted_index[token] = [title]

    # Save the inverted index to a file for later retrieval
    with open("research_articles_index.json", "w") as index_file:  # KINDLY INCLUDE THE CORRECT PATH
        json.dump(inverted_index, index_file)

    # Load the inverted index from the file
    with open("research_articles_index.json", "r") as index_file:  # please confirm the inverted index file path
        inverted_index = json.load(index_file)

    # Function for text preprocessing
    def preprocess_text(text):
        # Tokenize the text
        tokens = text.split()

        return tokens

    # Function to retrieve documents for a query
    def retrieve_documents(query, inverted_index):
        query = preprocess_text(query)

        # Initialize an empty list for result documents
        result_documents = set()  # Use a set to automatically remove duplicates

        for term in query:
            if term in inverted_index:
                result_documents.update(inverted_index[term])

        return list(result_documents), query


    # Prompt the user for a query
    user_query = input("Enter your query: ")

    # Retrieve documents for the user's query
    results, query_terms = retrieve_documents(user_query, inverted_index)

    # Check if there are matching documents
    if results:
        print("Search query terms", ' '.join(query_terms))

        # Import necessary libraries for text vectorization and similarity computation
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import linear_kernel

        matching_documents = research_data[research_data['Title'].isin(results)]

        if not matching_documents.empty:
            # Calculate the TF-IDF scores for documents
            document_content = research_data['Title'].tolist()  # Extract the content from the CSV
            vectorizer = TfidfVectorizer()
            tfidf_matrix = vectorizer.fit_transform(document_content)

            # Calculate the TF-IDF scores for the user's query
            query_tfidf = vectorizer.transform([user_query])

            # Compute cosine similarities between the query and each document
            cosine_similarities = linear_kernel(query_tfidf, tfidf_matrix)

            # Creating a list of document scores and sorting it
            document_scores = list(enumerate(cosine_similarities[0]))
            document_scores.sort(key=lambda x: x[1], reverse=True)

            # Display search results with TF-IDF scores for the query
            print("Search Results with Tf-IDF Scores for Query:")
            # Iterate through the sorted document scores and displaying relevant information
            for i, (index, score) in enumerate(document_scores):
                # Extract title and content row
                title = document_content[index]
                content_row = matching_documents[matching_documents['Title'] == title]['Content']

                # Display matching documents for research articles
                if not content_row.empty:
                    content = content_row.values[0]
                    tfidf_score = score

                    print(f"Document Title: {title}")
                    print(f"Document Content: {content}")
                    print(f"TF-IDF Score: {tfidf_score}")
                    print("------")
                # feedback when no matching documents are found
                else:
                    print(f"No matching documents found for title: {title}")
            # feedback when no matching documents are found
            else:
                print("No matching documents found for the query.")

    # feedback when no matching documents are found
    else:
        print("No matching documents found for the query.")
