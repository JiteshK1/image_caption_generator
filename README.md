# image_caption_generator
An Image captioning web application combines the power of React.js for front-end, Flask and Node.js for back-end, utilizing the MERN stack. Users can upload images and instantly receive automatic captions. Authenticated users have access to extra features like translating captions and text-to-speech functionality.

# Home Page

![Screenshot (118)](https://github.com/bhushan2311/image_caption_generator/assets/102802326/bc542a9e-f9c6-46b7-b10e-74a6db4dc2bc)


# Upload page (Guest)

UI for the users who aren't Logged-in

![Screenshot (126)](https://github.com/bhushan2311/image_caption_generator/assets/102802326/9af5e459-a48c-448b-86c9-7241a6df7126)


# Upload page (Logged In)
UI for the users who are Logged-in

![Screenshot (121)](https://github.com/bhushan2311/image_caption_generator/assets/102802326/607c9dbd-16ff-435d-9e16-d3136113ea0a)



# Result Page

The users can get the generated captions on this page. To get the access to the features of text-to-speech and caption translation they need to get authenticate by logging in. The link is provided below the generated caption which get navigated to the login page.

![Screenshot (142)](https://github.com/bhushan2311/image_caption_generator/assets/102802326/231a0c19-7c11-4b84-bdb0-0b83daa83a3e)


.


After successfully Logging in, the user can see the text-to-speech and translation feature as shown in the image below. For text-to-speech, react-speech-kit library is used. For translation, Translation Api provided by RapidApi has been used. User can chooses its preferred language from the dropdown to translate caption.

![Screenshot (143)](https://github.com/bhushan2311/image_caption_generator/assets/102802326/6d488652-16ba-48c4-849c-84b02d81a26c)



![Screenshot (144)](https://github.com/bhushan2311/image_caption_generator/assets/102802326/0241be1b-8685-4233-b43a-0cf35a2501be)




![Screenshot (146)](https://github.com/bhushan2311/image_caption_generator/assets/102802326/ca1bba97-4fa7-4a41-9b73-3194aafa18dc)




![Screenshot (145)](https://github.com/bhushan2311/image_caption_generator/assets/102802326/90b82455-20ac-459a-8948-b464830186b0)






# Login Page

![Screenshot (119)](https://github.com/bhushan2311/image_caption_generator/assets/102802326/916f713d-acec-4789-a962-fa16f574acac)


# Signup page




![Screenshot (120)](https://github.com/bhushan2311/image_caption_generator/assets/102802326/4fc3a436-7803-4889-b595-2a19ca5c7b44)



## How to Run the Project

Follow these steps to run the project on your local machine:

1. **Frontend (React):**
   - Open a terminal (Terminal-1).
   - Navigate to the 'frontend' directory using 'cd frontend/'.
   - Run the following command to start the React development server:
     ```bash
     npm install
     npm run start
     ```

2. **Backend (Flask):**
   - Open another terminal (Terminal-2).
   - Navigate to the 'server' directory using 'cd server/'.
   - Run the following command to start the Flask server:
     ```bash
     python app.py
     ```

3. **MongoDB Setup (Optional):**
   - If you want to use the login/signup, text-to-speech, and translation features, you'll need to set up MongoDB.
   - Open a third terminal (Terminal-3).
   - Navigate to the 'backend' directory using 'cd backend/'.
   - Start the Node.js server:
     ```bash
     node app.js
     ```
   - Open a fourth terminal (Terminal-4).
   - Start the MongoDB server:
     ```bash
     mongod
     ```
   - Open a fifth terminal (Terminal-5).
     ```bash
     mongo
     ```

4. **Access the Application:**
   - Open your web browser and go to [http://localhost:3000](http://localhost:3000).
