#!/usr/bin/env python
# coding: utf-8

# In[10]:


import face_recognition as fr
import cv2
import os
import matplotlib.pyplot as plt


# In[11]:


KNOWN_FACES = "known"
UNKNOWN_FACES = "unknown"
TOLERANCE = 0.5
FRAME_THICKNESS = 3
FONT_THICKNESS = 2
MODEL = 'hog'


# In[12]:


print("loading known faces")
known_faces=[]
known_names=[]


# In[13]:


for name in os.listdir(KNOWN_FACES):
    for filename in os.listdir(f"{KNOWN_FACES}/{name}"):
        try:
            image = fr.load_image_file(f"{KNOWN_FACES}/{name}/{filename}")
            encoding = fr.face_encodings(image)[0]
            known_faces.append(encoding)
            known_names.append(name)
        except:
            print("following file cannot be accessed ",filename)
        #print(encoding)


# In[14]:


print("processing unknown faces")
for filename in os.listdir (UNKNOWN_FACES):
    print(filename)
    image =  fr.load_image_file(f"{UNKNOWN_FACES}/{filename}")
    locations = fr.api.face_locations(image,number_of_times_to_upsample=1,model=MODEL)
    encodings = fr.face_encodings(image,locations)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    for face_encoding, face_location in zip(encodings,locations):
        results = fr.compare_faces(known_faces,face_encoding, TOLERANCE)
        match = None
        if True in results:
            match = known_names[results.index(True)]
            print(f"Match found: {match}")
            
            tlc = (face_location[3], face_location[0])
            brc = (face_location[1], face_location[2])
            Color = [0,255,0]
            cv2.rectangle(image,tlc,brc,(255,0,0),3)
            
            tlc = (face_location[3], face_location[2])
            brc = (face_location[1], face_location[2]+22)
            cv2.rectangle(image,tlc,brc,(255,0,0),cv2.FILLED)
            cv2.putText(image,match,(face_location[3]+10, face_location[0]+15), cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,0,0),FONT_THICKNESS)
    #cv2.imshow(filename,image)
    #cv2.waitKey(5000)
    plt.axis("off")
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.show()
            


# In[ ]:




