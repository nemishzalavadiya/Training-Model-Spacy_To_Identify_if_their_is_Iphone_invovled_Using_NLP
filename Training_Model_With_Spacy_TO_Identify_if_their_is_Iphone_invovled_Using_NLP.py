#!/usr/bin/env python
# coding: utf-8

# In[114]:


from spacy.matcher import Matcher
import spacy
import random
nlp=spacy.load('en_core_web_sm')


# In[115]:


TEXTS=['How to preorder the iPhone X',
 'iPhone X is coming',
 'Should I pay $1,000 for the iPhone X?',
 'The iPhone 8 reviews are here',
 'Your iPhone goes up to 11 today',
 'I need a new phone! Any tips?']
TEXTS_TEST_DATA=['Apple is slowing down the iPhone 8 and iPhone X - how to stop it',
 "I finally understand what the iPhone X 'notch' is for",
 'Everything you need to know about the Samsung Galaxy S9',
 'Looking to compare iPad models? Here’s how the 2018 lineup stacks up',
 'The iPhone 8 and iPhone 8 Plus are smartphones designed, developed, and marketed by Apple',
 'what is the cheapest ipad, especially ipad pro???',
 'Samsung Galaxy is a series of mobile computing devices designed, manufactured and marketed by Samsung Electronics with iPhone 5']
doc = list(nlp.pipe(TEXTS))
matcher = Matcher(nlp.vocab)


# In[116]:


# Two tokens whose lowercase forms match 'iphone' and 'x'
pattern1 = [{'LOWER': 'iphone'}, {'LOWER': 'x'}]

# Token whose lowercase form matches 'iphone' and an optional digit
pattern2 = [{'LOWER': 'iphone'}, {'IS_DIGIT': True,"OP":"?"}]

# Add patterns to the matcher
matcher.add('GADGET', None, pattern1, pattern2)


# In[117]:


# Create a Doc object for each text in TEXTS which creates a context for GADGETS if any
for doc in list(nlp.pipe(TEXTS)):
    # Find the matches in the doc
    matches = matcher(doc)
    
    # Get a list of (start, end, label) tuples of matches in the text
    entities = [(start, end, 'GADGET') for match_id, start, end in matches]
    print(doc.text, entities) 


# In[124]:


#Creation of Training Data
TRAINING_DATA = []

# Create a Doc object for each text in TEXTS
for doc in nlp.pipe(TEXTS):
    # Match on the doc and create a list of matched spans
    spans = [doc[start:end] for match_id, start, end in matcher(doc)]
    # Get (start character, end character, label) tuples of matches
    entities=[]
    print(spans)
    for span in spans:
        for i in entities:
            print(i)
            if i[0]==span.start_char:
                if i[1]<span.end_char:
                    del[entities[entities.index(i)]]
                else:
                    break
        else:
            entities.append((span.start_char, span.end_char, 'GADGET'))
    # Format the matches as a (doc.text, entities) tuple
    training_example = (doc.text, {'entities': entities})
    # Append the example to the training data
    TRAINING_DATA.append(training_example)
    
#Before you train a model with the data, you always want to double-check that 
#your matcher didn't identify any false positives. 
#But that process is still much faster than doing everything manually.    
print(*TRAINING_DATA, sep='\n')  


# In[125]:


# Create a blank 'en' model
nlp = spacy.blank('en')


# In[126]:


# Create a new entity recognizer and add it to the pipeline
ner = nlp.create_pipe('ner')
nlp.add_pipe(ner)


# In[127]:


# Add the label 'GADGET' to the entity recognizer
ner.add_label('GADGET')


# In[128]:


# Start the training - initialize weigths to random
nlp.begin_training()

# Loop for 10 iterations
for itn in range(10):
    # Shuffle the training data
    random.shuffle(TRAINING_DATA)
    losses = {}
    for batch in spacy.util.minibatch(TRAINING_DATA, size=2):
        texts = [text for text, entities in batch]
        annotations = [entities for text, entities in batch]                
        nlp.update(texts, annotations, losses=losses)
        print(losses)


# In[129]:


# Process each text in TEXTS_TEST_DATA
for doc in nlp.pipe(TEXTS_TEST_DATA):
    # Print the document text and entitites
    print(doc.text)
    print(doc.ents, '\n\n')


# In[ ]:




