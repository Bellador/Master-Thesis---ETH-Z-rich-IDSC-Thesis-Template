# Detection of Recreation Activities in Social Media

Full report is availble [here](https://drive.google.com/file/d/1LUaHJSoy5LKM-fSKQK6TyDGgtbVmRV10/view?usp=sharing)

In this master thesis a machine learning model was developed to automatically predict seven different nature based recreation activites.
These were:
- Walking
- Hiking
- Jogging
- Biking
- Dog Walking
- Horseback Riding
- Picnic

The LinearSVM model was trained on georeferenced, manually anotated Instagram posts. Image features were extracted by using the Google Cloud Vision API to detect text labels for contained visual elements. Afterwards the user generated text (post title, description and tags) and the text labels for the visual elements of the images were combined and used as features for a bag-of-words model. The model was tested not only on Instagram posts (which it was trained on) but also Flickr posts. The model achieved a precision of 0.75 across both Social Media Platforms as well as all seven classes. As seen in the figure below, the class specific precision is the lowest for the 'jogging' (0.62 Instagram, 0.12 Flickr) and 'picnic' (0.48 Instagram, 0.17 Flickr). Jogging was hard to differentiate from other similar classes such as 'walking' and 'hiking'. Picnic was a entirely a hard concept to grasp, since it needed to involve food & outdoor. Restaurant food images were responsible for most false classifications. All other classes performed much better.

<p align="center">
  <img src="https://github.com/Bellador/RecreationDetection/blob/master/img/M2_class_precision_unseen_data_w_vision_labels-1.png" width="650" title="class specific model precision">
</p>

It was proven that the combination of text and image data resultat in better performing model. 
A detailed workflow is visible below, alongside with maps showing the detected spatial distributions for the seven nature based recreations activities in the Canton of Zug, Switzerland.

![workflow](https://github.com/Bellador/RecreationDetection/blob/master/img/ML_text_data_visualization_cropped-1.png)
![map1](https://github.com/Bellador/RecreationDetection/blob/master/img/map_cluster_1-1.png)
![map2](https://github.com/Bellador/RecreationDetection/blob/master/img/map_cluster_2-1.png)
