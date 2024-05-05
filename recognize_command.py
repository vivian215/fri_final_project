from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import speech_to_text as speech

#determines if input command is one of two valid commands
def recognize_command():
    count_vectorizer = CountVectorizer()

    command0 = "go in that direction move there pointing" #"go there"
    command1 = "what is what's this unfamiliar object pointing at" #"what is this"

    THRESHOLD = 0.5

    #-1 = neither, 0 = go in direction, 1 = detect object
    output_command = -1 

    #get input vocal command
    speech.main()
    if speech.finished_processing:
        input_command = speech.command
        count_vectorizer.fit([command0, command1, input_command])

        #determine which command it is using feature vectors
        matrix0 = count_vectorizer.transform([command0]).todense()
        matrix1 = count_vectorizer.transform([command1]).todense()
        command_matrix = count_vectorizer.transform([input_command]).todense()
        similarity0 = cosine_similarity(np.asarray(matrix0), np.asarray(command_matrix))[0][0]
        similarity1 = cosine_similarity(np.asarray(matrix1), np.asarray(command_matrix))[0][0]

        if similarity0 >= THRESHOLD and similarity1 >= THRESHOLD:
            output_command = 1 if similarity1 > similarity0 else 0
        elif similarity0 >= THRESHOLD:
            output_command = 0
        elif similarity1 >= THRESHOLD:
            output_command = 1

    return output_command