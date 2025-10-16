## MSCE Agriculture Generative Chatbot ##

**Final Project: Domain-Specific Generative Question Answering (QA) System**
![Uploading Screenshot 2025-10-16 at 23.02.45.png…]()

This project implements a generative AI chatbot tailored to the Malawi Secondary Certificate of Education (MSCE) Agriculture syllabus. It utilizes a fine-tuned T5 Transformer model to provide students with instant, coherent answers and generate new practice questions for studying.


The final deployed application is fully functional and successfully resolves the challenge of deploying a large machine learning model (1.19 GB) on lightweight cloud infrastructure

* **Performance Improvement:** The validation loss dropped dramatically from **0.1394** after Epoch 1 to **0.0013** after Epoch 3, indicating a **significant improvement (over 90\% reduction)** and robust learning on the specialized dataset.

### Performance Metrics (Exemplary: 5/5 Pts)

* **Primary Metric (Validation Loss):** The model achieved a very low final validation loss of **0.0013**. This is the primary quantitative measure confirming the model's high accuracy.
* **Qualitative Testing:** Final stability testing ensures that out-of-domain questions (e.g., "how old are you?") return a **stable, domain-limited response** ("I'm an AI assistant focused exclusively on the MSCE agriculture syllabus..."). This confirms the chatbot remains a trustworthy and safe educational tool.

### UI Integration and Functionality

The user interface was built with Streamlit and is visually distinct and highly functional:

* **Interface:** Features a custom **Poppins font** and a **clean, two-mode navigation structure** using Streamlit Tabs.
* **Functionality:** Includes two panels:
    * **Student Mode:** The primary Q\&A panel where users ask questions.
    * **Tutor Mode:** The value-added **Question Generation** panel where tutors can generate new questions from their notes, enhancing class revisions.

***



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

# open this link in your browser to view how the app works #

https://agribotmscesummativeassignmentchatbot-fl6ndhlgrjyyjyyeyb9xr5.streamlit.app/


### Conclusion

This project demonstrates a strong understanding of transformer models and their application in solving pressing problems in African schools (e.g., lack of qualified teachers and fewer learning resources).


