#python program to create the web application

# importing the libraries
import streamlit as st
import tensorflow as tf
import numpy as np

# function to run the crop model and predict
def model_prediction(test_img):
    # loading the model
    loaded_model = tf.keras.models.load_model('trained_Model_1.keras')
    
    # image processing
    # total processing done on the image: image -> image of size 128,128 pixels -> np array of shape (128, 128, 3) ->batch of images os shape (n, 128, 128, 3) , n = no. of images
    image = tf.keras.preprocessing.image.load_img(test_img, target_size=(128,128))
    img_arr = tf.keras.preprocessing.image.img_to_array(image)
    img_arr = np.array([img_arr])
    
    # making the predictions based on the model
    prediction = loaded_model.predict(img_arr)
    result_class_index = np.argmax(prediction)
    
    # returning the index of result class
    return result_class_index


# function to run the weed classification model and predict
def weed_model_prediction(test_img):
    # loading the model
    loaded_model = tf.keras.models.load_model('weed_types\weed_trained_Model.keras')
    
    # image processing
    # total processing done on the image: image -> image of size 128,128 pixels -> np array of shape (128, 128, 3) ->batch of images os shape (n, 128, 128, 3) , n = no. of images
    image = tf.keras.preprocessing.image.load_img(test_img, target_size=(128,128))
    img_arr = tf.keras.preprocessing.image.img_to_array(image)
    img_arr = np.array([img_arr])
    
    # making the predictions based on the model
    prediction = loaded_model.predict(img_arr)
    result_class_index = np.argmax(prediction)
    
    # returning the index of result class
    return result_class_index

# creating a sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Page of the Web App", ["Home", "About", "Crop Disease Recognition", "Type of Weed Recognition"])

# the home page
if(app_mode == "Home"):
    # header
    st.header("Weed Recognition and Crop Disease Detection System using Deep learning")
    st.subheader("Batch 1 : Group 2, Web Technology Project")
    
    #image on the home screen 
    image_path = 'home_page_img2.webp'
    st.image(image_path, use_column_width=True)
    
    # info about the website
    # st.markdown("Information regarding the website")
    st.markdown("""
    Welcome to the Plant Disease Recognition System! 🌿🔍
    
    Our aim is to help in identifying crop diseases efficiently. Upload an image of a plant, and our system will analyze it to detect any signs of diseases!

    ### How It Works
    1. **Upload Image:** Go to the **Disease Recognition** page and upload an image of a plant with suspected diseases.
    2. **Analysis:** Our system will process the image using advanced algorithms to identify potential diseases.
    3. **Results:** View the results and recommendations for further action.

    ### Why Choose Us?
    - **Accuracy:** Our system is major part accurate with Validation Accuracy being **93%** using machine learning techniques.
    - **User-Friendly:** Simple and intuitive interface for seamless user experience.
    - **Fast and Efficient:** Receive results in seconds, allowing for quick decision-making.

    ### Get Started
    Click on the **Disease Recognition** page in the sidebar to upload an image and check results.

    ### About Us
    Learn more about the our team on the **About** page.
    """)
    
    

#About Page
elif(app_mode=="About"):
    # header
    st.header("About")
    
    # Information about Dataset
    st.markdown("""
                #### About the Technique
                We used Convolutional Neural Network to train our model on training data, Convolutional neural networks, also known as CNNs, are a specific type of neural networks that are generally composed of the following layers:
                1. **Convolution layer (CONV):** The convolution layer (CONV) uses filters that perform convolution operations as it is scanning the input I with respect to its dimensions. Its hyperparameters include the filter size F and stride S.
                The resulting output O is called feature map or activation map.
                2. **Pooling (POOL):** The pooling layer (POOL) is a downsampling operation, typically applied after a convolution layer, which does some spatial invariance.
                3. **Connection Layer:** The fully connected layer (FC) operates on a flattened input where each input is connected to all neurons.
                #### About Crop Dataset
                This dataset is recreated using offline augmentation from the original dataset.The original dataset can be found on Kaggle and is contributed by **@vipoooool**(Samir Bhattarai).
                This dataset consists of about 24K rgb images of healthy and diseased crop leaves which is categorized into 13 different classes.The total dataset is divided into 80/20 ratio of training and validation set preserving the directory structure.
                A new directory containing 21 test images is created later for prediction purpose.
                #### Crop Dataset Content
                1. train (24055 images)
                2. test (21 images)
                3. validation (6014 images)
                #### About Weed Dataset
                This dataset majorly has been collected from roboflow and later structured into a train, test, and validation set by us.
                This dataset consists of about 1000 rgb images of different kinds of weeds which is categorized into 7 different classes.The total dataset is divided into 70/20/10 ratio of training, validation and testing set preserving the directory structure.
                #### Weed Dataset Content
                1. train (628 images)
                3. validation (244images)
                2. test (89 images)

                """)
    
    # Information about the team
    st.markdown("""
                ### About the Team
                1. Susmit Bahadkar: Deptartment of Information Technology at Vishwakarma Institute of Technolgy, Pune.
                2. Parth Bhalerao: Deptartment of Information Technology at Vishwakarma Institute of Technolgy, Pune.
                3. Abhilash Baviskar: Deptartment of Information Technology at Vishwakarma Institute of Technolgy, Pune.
                4. Jaywant Avhad: Deptartment of Information Technology at Vishwakarma Institute of Technolgy, Pune.
                5. Akash Chimkar: Deptartment of Information Technology at Vishwakarma Institute of Technolgy, Pune.
                """)
    
# Disease Recognition Page
elif(app_mode=="Crop Disease Recognition"):
    
    # mardown to tell about the model
    st.markdown("Our Model of Crop Disease Detection gives upto **91%** Validation Accuracy, and can recognize **13** different kinds of weeds.")
    
    # header
    st.header("Crop Disease Recognition")
    
    # uploading the image
    test_image = st.file_uploader("Choose an Image:")
    
    # show the image
    if(st.button("Show Image")):
        st.image(test_image,width=4,use_column_width=True)
    
    # Predict button
    if(st.button("Predict")):
        st.write("Our Prediction")
        result_class_index = model_prediction(test_image)
        #Reading Labels
        class_name = ['Apple___Cedar_apple_rust',
                    'Apple___healthy',
                    'Corn_(maize)___healthy',
                    'Grape___Black_rot',
                    'Grape___healthy',
                    'Potato___Early_blight',
                    'Potato___Late_blight',
                    'Potato___healthy',
                    'Strawberry___Leaf_scorch',
                    'Strawberry___healthy',
                    'Tomato___Bacterial_spot',
                    'Tomato___Early_blight',
                    'Tomato___healthy']
        st.success("Model is Predicting it's a {}".format(class_name[result_class_index]))
        

# weed classification page
elif(app_mode=="Type of Weed Recognition"):
    
    # markdown to tell about the model
    st.markdown("Our Model of Weed Recognition gives upto **90%** Training Accuracy **80%** Validation Accuracy, and can recognize **7** different kinds of weeds.")
    
    # header
    st.header("Upload an Image and see what kind of weed that is")
    
    # uploading the image
    test_image = st.file_uploader("Choose an Image:")
    
    # show the image
    if(st.button("Show Image")):
        st.image(test_image,width=4,use_column_width=True)
    
    # Predict button
    if(st.button("Predict")):
        st.write("Our Prediction")
        result_class_index = weed_model_prediction(test_image)
        #Reading Labels
        class_name = ['Ambrosia_Rag_weed',
                'Amsinkia_Chickpeas',
                'Cannabis_Marijuana',
                'Common_taraxacum_dandelion',
                'Erigeron_canadensis_horse_weed',
                'Otanthus_maritimus_Cotton_Weed',
                'Trianthema_portulacastrum_Pig_Weed']
        st.success("Model is Predicting it's a {}".format(class_name[result_class_index]))
        
        # display the image
        st.image(test_image,width=400)