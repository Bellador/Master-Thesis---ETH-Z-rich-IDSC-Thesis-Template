# Detection of Recreation Activities in Social Media

In this master thesis a machine learning model was developed to automatically predict seven different nature based recreation activites.
These were:
- Walking
- Hiking
- Jogging
- Biking
- Dog Walking
- Horseback Riding
- Picnic

The LinearSVM model was trained on georeferenced, manually anotated Instagram and Flickr posts. Image features were extracted by using the Google Cloud Vision API to detect text labels for contained visual elements. Afterwards the user generated text (post title, description and tags) and the text labels for the visual elements of the images were combined and used as features for a bag-of-words model. A detailed workflow is visible below, alongside with the detected spatial distributions for the seven nature based recreations activities in the Canton of Zug, Switzerland.

![workflow](https://github.com/Bellador/RecreationDetection/blob/master/img/ML_text_data_visualization_cropped.png)
![map1](https://github.com/Bellador/RecreationDetection/blob/master/img/map_cluster_1.png)
![map2](https://github.com/Bellador/RecreationDetection/blob/master/img/map_cluster_2.png)
