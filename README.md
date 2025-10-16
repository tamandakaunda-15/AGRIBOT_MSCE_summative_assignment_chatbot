## MSCE Agriculture Generative Chatbot ##
#Final Project: Domain-Specific Generative Question Answering (QA) System#

This project implements a generative AI chatbot tailored to the Malawi Secondary Certificate of Education (MSCE) Agriculture syllabus. It utilizes a fine-tuned T5 Transformer model to provide students with instant, coherent answers and generate new practice questions for studying.

The final deployed application is fully functional and successfully resolves the challenge of deploying a large machine learning model (1.19 GB) on lightweight cloud infrastructure





## Set Up process ##

1. Clone the Repository
```
git clone [https://github.com/tamandakaunda-15/AGRIBOT_MSCE_summative_assignment_chatbot.git](https://github.com/tamandakaunda-15/AGRIBOT_MSCE_summative_assignment_chatbot.git)
cd AGRIBOT_MSCE_summative_assignment_chatbot

```

2. SeTup Environment

```
conda create -n chatbot-env python=3.11
conda activate chatbot-env
pip install streamlit tensorflow transformers datasets pandas tf-keras

```

3. Run the App
```
streamlit run app.py
```
(Note: The first time you run this, the Hugging Face client will automatically download the 1.19 GB model file from the Hub ID specified inside app.py and cache it.)


OR

## open this link in your browser to view how the app works ##

https://agribotmscesummativeassignmentchatbot-fl6ndhlgrjyyjyyeyb9xr5.streamlit.app/


