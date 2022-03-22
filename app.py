#!/usr/bin/env python
# coding: utf-8

# In[1]:


from textblob import TextBlob
from flask import Flask, request, render_template
from transformers import pipeline


# In[2]:


app = Flask(__name__)
classifier = pipeline('sentiment-analysis', "mrm8488/bert-small-finetuned-squadv2")


# In[3]:


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        text = request.form.get("text")
        print(text)
        r1 = TextBlob(text).sentiment
        r2 = classifier(text)
        return(render_template("index.html", result1=r1, result2=r2))
    else:
        return(render_template("index.html", result1="2", result2="2"))


# In[ ]:


if __name__ == "__main__":
    app.run()


# In[ ]:




