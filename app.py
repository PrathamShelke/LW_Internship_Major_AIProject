import streamlit as st
import pywhatkit
import pyttsx3
import webbrowser
import smtplib
import os
import time
import plyer
import cv2
import cvzone
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import openai
import io
from PIL import Image
# Set OpenAI API key
openai.api_key = "enter api key"

def main():
    im = Image.open('lw.png')
    # Adding Image to web app and giving page title
    st.set_page_config(page_title="Pragati.AI", page_icon=im)
    st.title(":green[Menu-Driven Program]")

    menu_options = [
        "Open bash terminal with admin privileges",
        "Send WhatsApp message",
        "Convert Text into Speech on Speaker",
        "Open any website URL",
        "Send email using SMTP Server",
        "Renaming files",
        "Notification for drinking water",
        "Volume Control using Hand Gesture",
        "Summarizing Translated Text",
        "Generative AI using GOOGLE SERP",
        "Car Price Predicter",
        "Bank Account Management ",
        "Youtube Recommendation System",
        "Artificial Neural Network",
        "Convolutional Neural Network",
        "Text Editor using Tkinter",
        "Stock Prediction(RNN)",
        "Generate Image using Dalle"
    ]

    choice = st.sidebar.selectbox("Select an option", menu_options)

    if choice == menu_options[0]:
        option1()
    elif choice == menu_options[1]:
        option2()
    elif choice == menu_options[2]:
        option3()
    elif choice == menu_options[3]:
        option4()
    elif choice == menu_options[4]:
        option5()
    elif choice == menu_options[5]:
        option6()
    elif choice == menu_options[6]:
        option7()
    elif choice == menu_options[7]:
        option8()
    elif choice == menu_options[8]:
        option9()
    elif choice == menu_options[9]:
        option10()
    elif choice == menu_options[10]:
        option11()
    elif choice == menu_options[11]:
        option12()
    elif choice == menu_options[12]:
        option13()
    elif choice == menu_options[13]:
        option14()
    elif choice == menu_options[14]:
        option15()
    elif choice == menu_options[15]:
        option16()
    elif choice == menu_options[16]:
        option17()
    elif choice == menu_options[17]:
        option18()


def option1():
    st.header("Open bash terminal with admin privileges")
    st.write("You selected Option 1.")
    # Add your code for Option 1 here
    if st.button("Open Website"):
        url = "http://43.205.178.207"
        webbrowser.open(url)
        st.success("Website opened successfully!")

def option2():
    st.header("Send WhatsApp message")
    st.write("You selected Option 2.")
    # Add your code for Option 2 here
    no = st.text_input("Enter 10 Digit number")
    message = st.text_input("Enter message:")
    hr = st.number_input("Enter the hour you want to send message:", min_value=0, max_value=23, value=0)
    minutes = st.number_input("Enter the minutes you want to send message:", min_value=0, max_value=59, value=0)
    if st.button("Send Message"):
        if no and message:
            no = "+91" + no
            pywhatkit.sendwhatmsg(no, message, hr, minutes)
            st.success("Message sent successfully!")

def option3():
    st.header("Convert Text into Speech on Speaker")
    st.write("You selected Option 3.")
    # Add your code for Option 3 here
    message = st.text_input("Enter message to speak:")
    if st.button("Speak"):
        if message:
            myspeaker = pyttsx3.init()
            myspeaker.say(message)
            myspeaker.runAndWait()
            st.success("Text converted to speech!")

def option4():
    st.header("Open any website URL")
    st.write("You selected Option 4.")
    # Add your code for Option 4 here
    url = st.text_input("Enter the URL:")
    if st.button("Open Website"):
        if url:
            url = "https://" + url
            webbrowser.open(url)
            st.success("Website opened successfully!")

def option5():
    st.header("Send email using SMTP Server")
    st.write("You selected Option 5.")
    # Add your code for Option 5 here
    sender_email = st.text_input("Enter sender email:")
    sender_password = st.text_input("Enter appCode/Sender Password:", type="password")
    receiver_email = st.text_input("Enter receiver email:")
    subject = st.text_input("Enter Subject of email:")
    message = st.text_area("Enter message:")

    if st.button("Send Email"):
        if sender_email and sender_password and receiver_email and subject and message:
            def send_email(sender_email, sender_password, receiver_email, subject, message):
                # Connect to the SMTP server
                server = smtplib.SMTP('smtp.gmail.com', 587)
                server.starttls()
                server.login(sender_email, sender_password)

                # Create the email message
                email_message = f"Subject: {subject}\n\n{message}"

                # Send the email
                server.sendmail(sender_email, receiver_email, email_message)

                # Disconnect from the server
                server.quit()

                # Print a success message
                st.success("Email sent successfully!")

            send_email(sender_email, sender_password, receiver_email, subject, message)

def option6():
    st.header("Renaming files")
    st.write("You selected Option 6.")
    # Add your code for Option 6 here
    directory = 'G:/My Drive/Data_Math_6TO10/ncert_solutions_byjus_6_to_10/Math_6_ncert_solutions_byjus'
    prefix = 'New_Class6'

    if st.button("Rename Files"):
        def rename_files(directory, prefix):
            # Get a list of all files in the directory
            file_list = os.listdir(directory)

            # Iterate over each file in the list
            for file_name in file_list:
                # Generate the new file name by adding the prefix
                new_file_name = prefix + file_name

                # Construct the full paths for the old and new file names
                old_path = os.path.join(directory, file_name)
                new_path = os.path.join(directory, new_file_name)

                # Rename the file by moving it to the new path
                os.rename(old_path, new_path)

            st.success("File renaming complete!")

        rename_files(directory, prefix)

def option7():
    st.header("Notification for drinking water")
    st.write("You selected Option 7.")
    # Add your code for Option 7 here
    if st.button("Start Drinking Water Notification"):
        def send_notification(message):
            plyer.notification.notify(
                title="Drink Water!",
                message=message,
                timeout=10
            )

        def notify_water():
            send_notification("It's time to drink water!")
            time.sleep(60)

        notify_water()

def option8():
    st.header("VOLUME Control using Hand Movement")
    st.write("You selected Option 8.")
    # Add your code for Option 8 here
    if st.button("Volume Control"):
        import cv2
        import numpy as np
        from cvzone.HandTrackingModule import HandDetector
        import pyautogui

        detector = HandDetector(detectionCon=0.8)
        min_volume = 0
        max_volume = 100
        cap = cv2.VideoCapture(0)
        while True:
            success, frame = cap.read()
            if not success:
                break
            frame = cv2.flip(frame, 1)
            hands, frame = detector.findHands(frame)
            if hands:
                for hand in hands:
                    landmarks = hand["lmList"]
                    bbox = hand["bbox"]
                    thumb_tip = landmarks[4]
                    index_tip = landmarks[8]
                    thumb_index_distance = np.linalg.norm(np.subtract(thumb_tip, index_tip))
                    volume = np.interp(thumb_index_distance, [20, 200], [min_volume, max_volume])
                    volume = int(max(min(volume, max_volume), min_volume))
                    pyautogui.press('volumedown') if volume < 50 else pyautogui.press('volumeup')
                    cv2.putText(frame, f"Volume: {volume}%", (bbox[0], bbox[1] - 20),
                                cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 2)
            cv2.imshow("Volume Control", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()

def option9():
    import streamlit as st
    from googletrans import Translator
    import pyttsx3
    from transformers import BartTokenizer, BartForConditionalGeneration

    st.header("Translation and Summarization of English to any other language")
    st.write("You selected Option 9.")
    # Add your code for Option 9 here
    source_text = st.text_input("Enter text:")
    source_language = "en"  # English
    target_language = st.text_input("In which language you want to convert (e.g., fr for French):")
    if st.button("Translate and Summarize"):
        if source_text and target_language:
            translator = Translator()
            translated_text = translator.translate(source_text, src=source_language, dest=target_language).text
            st.write(f"Translation: {translated_text}")

            prompt = f"Summarize the following text: {translated_text}"
            response = openai.Completion.create(
                engine="text-davinci-003",
                prompt=prompt,
                max_tokens=150,
                temperature=0.7
            )
            summary = response.choices[0].text.strip()

            st.write("Summary:")
            st.write(summary)
            # Text-to-speech for the summary
            engine = pyttsx3.init()
            engine.setProperty('rate', 150)
            engine.setProperty('volume', 0.8)
            engine.say(summary)
            engine.runAndWait()




def option10():
    st.header("GOOGLE Search + Chatgpt Api")
    source_text = st.text_input("Enter text:")
    import openai
    if st.button("Result"):
        if source_text:
            import os
            from langchain.llms import OpenAI
            os.environ['SERPAPI_API_KEY'] = "enter serpapi key"
            mykey = "enterkey"
            from langchain.agents import load_tools
            mytools = load_tools(tool_names=["serpapi"])
            myllm = OpenAI(temperature=0, openai_api_key=mykey)
            from langchain.agents import initialize_agent
            from langchain.agents import AgentType
            from langchain.agents import agent_types
            my_google_chain = initialize_agent(
                tools=mytools,
                llm=myllm,
                agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                # Using Zero shot agent(Chain of thought) for Google search tool,llm model openai
                verbose=True
            )
            output=my_google_chain.run(source_text)
            st.write(output)
def option11():
    import pandas as pd
    from sklearn.linear_model import LinearRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error

    # Load the dataset
    data = pd.read_csv('car_data.csv')

    # Split the data into features (X) and target variable (y)
    X = data[['mileage', 'age']]
    y = data['price']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print('Mean Squared Error:', mse)

    import streamlit as st

    # Load the trained model
    model = LinearRegression()
    model.fit(X, y)

    # Function to predict the car price based on user inputs
    def predict_price(mileage, age):
        features = [[mileage, age]]
        price = model.predict(features)
        return price[0]

    # Streamlit application

    st.title("Car Price Prediction")
    st.write("Enter the car's mileage and age to predict its price.")

    # User input fields
    mileage = st.number_input("Mileage (in kilometers)")
    age = st.number_input("Age (in years)")

    if st.button("Predict"):
        price = predict_price(mileage, age)
        st.success(f"The predicted price of the car is {price:.2f}.")

def option12():
    class BankAccount:
        def __init__(self, name, account_number, account_type):
            self.name = name
            self.account_number = account_number
            self.account_type = account_type
            self.balance = 0


    st.title("Bank Account App")

    session = st.session_state

    if 'balance' not in session:
        session.balance = 0

    name = st.text_input("Enter your name")
    account_number = st.text_input("Enter your account number")
    account_type = st.text_input("Enter your account type")

    account = BankAccount(name, account_number, account_type)

    st.subheader("Bank Account Menu")

    with st.form("deposit_form"):
        deposit_amount = st.number_input("Enter the amount to deposit", step=0.01, min_value=0.01)
        deposit_submit_button = st.form_submit_button("Submit Deposit")

    if deposit_submit_button:
        account.balance += deposit_amount
        session.balance = account.balance
        st.write("Deposit successful. New balance: {:.2f}".format(account.balance))

    with st.form("withdraw_form"):
        withdraw_amount = st.number_input("Enter the amount to withdraw", step=0.01, min_value=0.01)
        withdraw_submit_button = st.form_submit_button("Submit Withdrawal")

    if withdraw_submit_button:
        if session.balance >= withdraw_amount:
            session.balance -= withdraw_amount
            account.balance = session.balance
            st.write("Withdrawal successful. New balance: {:.2f}".format(account.balance))
        else:
            st.error("Insufficient balance. Withdrawal failed.")

    if st.button("Display Information"):
        st.write("Name:", account.name)
        st.write("Account Number:", account.account_number)
        st.write("Account Type:", account.account_type)
        st.write("Balance:", session.balance)

    if st.button("Quit"):
        st.write("Thank you for using our banking services. Goodbye!")
def option13():
    import streamlit as st
    from googleapiclient.discovery import build
    import random

    # API key for accessing YouTube Data API
    API_KEY = "enter yt api key"

    @st.cache
    def get_video_recommendations(query):
        # Build the YouTube Data API service
        youtube = build('youtube', 'v3', developerKey=API_KEY)

        # Make a search request based on the user query
        search_response = youtube.search().list(
            q=query,
            type='video',
            part='id,snippet',
            maxResults=10
        ).execute()

        video_results = search_response.get('items', [])

        # Shuffle the video results
        random.shuffle(video_results)

        # Return a list of dictionaries containing video details
        video_recommendations = []
        for video_result in video_results:
            video_title = video_result['snippet']['title']
            channel_name = video_result['snippet']['channelTitle']
            video_id = video_result['id']['videoId']
            video_url = f"https://www.youtube.com/watch?v={video_id}"
            thumbnail_url = video_result['snippet']['thumbnails']['medium']['url']

            video_recommendations.append({
                'title': video_title,
                'channel': channel_name,
                'url': video_url,
                'thumbnail': thumbnail_url
            })

        return video_recommendations

    # Streamlit application
    st.title("YouTube Video Recommender")

    user_query = st.text_input("Enter your query")

    if st.button("Search"):
        recommendations = get_video_recommendations(user_query)
        st.header("Recommended videos:")
        for video in recommendations:
            st.subheader(video['title'])
            st.text(f"Channel: {video['channel']}")
            st.image(video['thumbnail'], use_column_width=True)
            st.write(f"URL: [Watch video]({video['url']})")
            st.markdown("---")
def option14():
    import streamlit as st
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.neural_network import MLPClassifier
    from sklearn.metrics import accuracy_score

    # Load the Iris dataset
    def load_data():
        iris_data = pd.read_csv("iris.csv")
        return iris_data

    # Prepare the data for training
    def prepare_data(iris_data):
        X = iris_data.iloc[:, :-1].values
        y = iris_data.iloc[:, -1].values

        # Encode the species names to integers
        class_mapping = {'Setosa': 0, 'Versicolor': 1, 'Virginica': 2}
        y = np.array([class_mapping[species] for species in y])

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Standardize the features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        return X_train, X_test, y_train, y_test, scaler

    # Train the neural network model
    def train_model(X_train, y_train):
        model = MLPClassifier(hidden_layer_sizes=(50,), max_iter=1000, random_state=42)
        model.fit(X_train, y_train)
        return model

    # Main function to run the Streamlit web app
    def main():
        st.title(":orange[Iris Flower Species Classification with ANN]")

        # Load the data
        iris_data = load_data()

        # Prepare the data for training
        X_train, X_test, y_train, y_test, scaler = prepare_data(iris_data)

        # Train the model
        model = train_model(X_train, y_train)

        # Show accuracy on the training set
        train_predictions = model.predict(X_train)
        train_accuracy = accuracy_score(y_train, train_predictions)
        #st.write(f"Training Accuracy: {train_accuracy:.2f}")

        # Show accuracy on the test set
        test_predictions = model.predict(X_test)
        test_accuracy = accuracy_score(y_test, test_predictions)
        #st.write(f"Test Accuracy: {test_accuracy:.2f}")

        # Input form for user to enter petal and sepal measurements
        st.header(":red[Enter Petal and Sepal Measurements]")
        sepal_length = st.number_input("Sepal Length", min_value=0.1, max_value=10.0, step=0.1)
        sepal_width = st.number_input("Sepal Width", min_value=0.1, max_value=10.0, step=0.1)
        petal_length = st.number_input("Petal Length", min_value=0.1, max_value=10.0, step=0.1)
        petal_width = st.number_input("Petal Width", min_value=0.1, max_value=10.0, step=0.1)

        # Make predictions based on user inputs
        user_input = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
        user_input_scaled = scaler.transform(user_input)
        prediction = model.predict(user_input_scaled)[0]
        species_mapping = {0: 'Setosa', 1: 'Versicolor', 2: 'Virginica'}
        predicted_species = species_mapping[prediction]

        # Display the predicted species
        st.header("Predicted Iris Species")
        st.write(predicted_species)
        if predicted_species=="Setosa":
            st.image("setosa.jpg", caption="Setosa", use_column_width=True)
        elif predicted_species=="Versicolor":
            st.image("versicolor.jpg", caption="Versicolor", use_column_width=True)
        else:
            st.image("virginica.jpg", caption="Virginica", use_column_width=True)

    if __name__ == "__main__":
        main()
def option15():
    import streamlit as st
    import keras.utils as image
    import numpy as np
    from keras.models import load_model
    from gtts import gTTS
    from io import BytesIO

    # Load the pre-trained model
    model = load_model("model.h5")
    import pyttsx3

    # Initialize the text-to-speech engine
    engine = pyttsx3.init()

    # Define a function for speech synthesis
    def speak(text):
        engine.say(text)
        engine.runAndWait()
    # Function to perform prediction and speak the result
    def predict_and_speak(image_path):

        img = image.load_img(image_path, target_size=(64, 64))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = x / 255.0

        prediction = model.predict(x)
        if prediction[0][0] <= 0.5:
            result = "cat"
        else:
            result = "dog"

        # Convert the result to speech and play it
        speak(f"This is a {result}")

    # Streamlit app
    def main():
        st.title(":orange[Image Classification using CNN]")

        st.write("Upload an image and let the app classify it!")

        # File uploader for image
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            # Display the uploaded image
            image_path = BytesIO(uploaded_file.read())
            st.image(image_path, caption="Uploaded Image", use_column_width=True)

            # Perform prediction and speak the result
            with st.spinner("Analyzing the image..."):
                predict_and_speak(image_path)

    if __name__ == "__main__":
        main()
def option16():
    import streamlit as st
    import subprocess

    def main():
        st.title(":blue[NoteIt]")
        # Run the Tkinter text editor as a subprocess
        if st.button("Click to Open NoteIt"):
            subprocess.Popen(["python", "text_editor.py"])

    if __name__ == "__main__":
        main()
def option17():
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.preprocessing import MinMaxScaler
    import streamlit as st

    # Load the stock data (replace 'google.csv' with your data file)
    data = pd.read_csv('google.csv')

    # Use 'Close' price for prediction
    data = data[['Close']]
    dataset = data.values
    dataset = dataset.astype('float32')

    # Normalize the dataset
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(dataset)

    # Split data into training and testing sets
    train_size = int(len(dataset) * 0.8)
    test_size = len(dataset) - train_size
    train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]

    # Create sequences for time series data
    def create_sequences(data, seq_length):
        sequences = []
        for i in range(len(data) - seq_length):
            sequence = data[i:i + seq_length]
            sequences.append(sequence)
        return np.array(sequences)

    seq_length = 10  # You can experiment with different sequence lengths
    X_train = create_sequences(train, seq_length)
    X_test = create_sequences(test, seq_length)

    # Load the pre-trained model
    from tensorflow.keras.models import load_model
    model = load_model("StockPredcition.h5")

    # Make predictions
    train_predict = model.predict(X_train)
    test_predict = model.predict(X_test)

    # Inverse transform the predictions
    train_predict = scaler.inverse_transform(train_predict)
    test_predict = scaler.inverse_transform(test_predict)

    # Prepare the data for plotting
    train_predict_plot = np.empty_like(dataset)
    train_predict_plot[:, :] = np.nan
    train_predict_plot[seq_length:len(train_predict) + seq_length, :] = train_predict

    test_predict_plot = np.empty_like(dataset)
    test_predict_plot[:, :] = np.nan
    test_predict_plot[len(train_predict) + (seq_length * 2) - 1:len(dataset) - 1, :] = test_predict

    # Streamlit app
    st.title(':orange[Stock Price Prediction using RNN]')
    st.write('Simple Stock Price Prediction using Recurrent Neural Network (RNN)')

    # Plot the results
    if st.button("Show Google Stock Trends"):

        plt.figure(figsize=(12, 6))
        plt.plot(data.index, scaler.inverse_transform(dataset), label='Actual')
        plt.plot(data.index[seq_length:len(train_predict) + seq_length],
                 train_predict_plot[seq_length:len(train_predict) + seq_length], label='Train Predicted')
        plt.plot(data.index[len(train_predict) + seq_length + 1:], test_predict_plot[len(train_predict) + seq_length + 1:],
                 label='Test Predicted')
        plt.xlabel('Date')
        plt.ylabel('Stock Price')
        plt.title('Stock Price Prediction using RNN')
        plt.legend()

        # Display the plot in Streamlit
        st.pyplot(plt)
def option18():
    import streamlit as st
    import openai
    import requests
    from PIL import Image
    import matplotlib.pyplot as plt

    # Replace the following line with your OpenAI API key
    openai.api_key = "enter api key"

    def generate_image(text):
        res = openai.Image.create(
            prompt=text,
            n=1,
            size="256x256",
        )
        return res["data"][0]["url"]

    def main():
        st.title(":blue[Text to Image Generation]")

        # User input for the prompt
        prompt = st.text_input("Enter your prompt:")

        # Generate image when the user clicks the "Generate" button
        if st.button("Generate"):
            try:
                image_url = generate_image(prompt)
                img_data = requests.get(image_url).content
                with open("generated_image.png", "wb") as img_file:
                    img_file.write(img_data)

                img = Image.open("generated_image.png")
                st.image(img, use_column_width=True)
            except Exception as e:
                st.error("Error: " + str(e))

    if __name__ == "__main__":
        main()


if __name__ == "__main__":
    main()
