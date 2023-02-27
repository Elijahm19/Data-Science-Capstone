ABSTRACT-
With the rise of artificial intelligence and the advancements in technology, the need for proper traffic sign recognition is important for aiding driver awareness of speed limits and other road signs. With autonomous, self-driving cars, it is very vital for the program to recognize different traffic signals to avoid any catastrophic accidents. The goal of our project is to analyze various traffic signs/signals and categorize them for autonomous vehicles to recognize signs effectively in order to increase road safety. Traffic sign recognition (TSR) can improve road safety by providing real-time information to drivers about their surroundings, such as speed limits, stop signs, and road conditions. Also, TSR may impact drivers’ behavior since they are aware of their surroundings due to the real-time feedback about their speed and compliance with traffic laws. Very likely in the future, the amount of self-driving cars and cars with advanced technology will rise drastically, so this technology has an essential real-world use. In summary, traffic sign recognition systems are a technology that can help with improving road safety, enhancing navigation, improving traffic flow, enhancing autonomous driving, and improving driver behavior.
In this project a number of road sign images will be processed using convolutional neural networks (CNN). Since this is a multi-class classification problem, the results of the processed images will be fed into classification machine learning models such as Random Forest, Gaussian Mixture Model, and One-vs-All multi-class Logistic Regression. The models will classify the images into four classes: traffic light, stop sign, speed limit, or crosswalk. Models will be evaluated and deployment recommendations will be provided.
____________________________________________________________________________________________________________________________________________________________
Progress report-

Business Understanding

1.1 Business Problem

As artificial intelligence and technology continue to advance, it has become increasingly crucial to ensure proper traffic sign recognition to enhance driver awareness of speed limits and other road signs. With the advent of autonomous, self-driving cars, the importance of accurate traffic signal recognition cannot be overstated, as any failure in this regard could lead to catastrophic accidents. Our project aims to analyze various traffic signs and signals, categorize them, and develop effective recognition systems for autonomous vehicles, thereby significantly increasing road safety. Real-time traffic sign recognition can provide drivers with essential information about their surroundings, such as speed limits, stop signs, and road conditions, thereby contributing to improved road safety. Additionally, such systems can positively impact driver behavior by providing real-time feedback on speed and compliance with traffic laws. Given the projected rise in self-driving cars and cars with advanced technology, traffic sign recognition technology has become an essential real-world application. 

1.2 Dataset 

The original dataset is split into two folders: one that contains the png images, and another that includes image annotations as xml files. There are 877 images with four distinct classes for the objective of road sign detection. The classes are Traffic light, Stop signs, Speed limits and Crosswalks. After reading information from both folders, the dataset is converted into a dataframe. The data frame consists of 877 rows, a row for each image and 7 descriptive features and one class feature. The columns are: filename, width, height, class label, xmin, ymin, xmax, and ymax. The filename attribute includes the actual image. Class feature shows the class that image belongs to. Width and Height represent image measurements, while the rest of the features display the bounding box coordinates.

1.3 Proposed Analytics Solution

The objective of this project is to process a set of road sign images using convolutional neural networks (CNNs). Given that this is a multi-class classification problem, the output from the processed images will be fed into machine learning models for classification, including Random Forest, Gaussian Mixture Model, and One-vs-All multi-class Logistic Regression. The aim is to classify the images into four classes: traffic light, stop sign, speed limit, or crosswalk. Once the models are trained, they will be evaluated, and recommendations for deployment will be provided. By leveraging CNNs and machine learning models, this project seeks to achieve efficient and accurate classification of road signs.

For professional implementation of this project, several management tools are utilized. Trello is a visual collaboration tool that gives teams a shared perspective on any project. It provides us a shared space to organize and collaborate. We are using Trello to set project goals, guidelines, and deadlines. Additionally, we are using Slack to communicate, share information, ask questions and clarifications about project details. Besides, we are using ML flow, which is a platform that helps manage end-to-end machine learning lifecycle. It provides a suite of tools to track experiments, package code into reproducible runs, and share and deploy models.Additionally, we are using ARCTIC, Georgia State’s advanced computing technology. With the use of ARCTIC we are able to satisfy our high computational needs in regards to running our models efficiently.

 Data Exploration and Preprocessing
 
Firstly, we visualized our images and annotations into a dataframe, then we converted our dataframe to a CSV file to continue our exploratory data analysis (EDA). During our analysis we used “value_counts'' to return the number of images in each class. The results indicated that the speed limit class showed up for 75% of the data, with the remaining percentage evenly shared amongst the other classes.


  2.1 Data Visualization

Figure 1.  Bar Plot of Classes

![image](https://user-images.githubusercontent.com/47839751/221710107-3f3fd09f-efbf-470a-908a-3339e47b9068.png)

2.1.1 Bar Plot of Classes

We used a bar chart to visualize the number of instances in each class. The bar chart reflected the findings of value_counts. It is evident that the speed limit label is dominating while the other labels are underrepresented. This calls for data augmentation to balance classes before applying classification machine learning models to avoid model bias. 

2.1.2 Average Images

The average images represent the "typical" image in the dataset. It shows what the objects in the images have in common, such as color, texture, and shape. Any variations among the images in the dataset will be represented by the deviations from the average image. In this case, the average image is mostly bluish gray, which does not reflect any specific part of the actual class. Additionally, the quality of the image shows that input images are not very clear.

Figure 2. Speed Limit Average Image

![image](https://user-images.githubusercontent.com/47839751/221708908-a001f4d1-6fd0-4faa-b851-81f25c4958c4.png)

A visualization of the speed limit average image

Figure 3. Stop Sign Average Image

![image](https://user-images.githubusercontent.com/47839751/221708984-d14084f7-40af-455d-b0ab-1f621527b6d9.png)

A visualization of the average stop sign

Figure 4. Crosswalk Average Image

![image](https://user-images.githubusercontent.com/47839751/221709038-fbd2c92a-7828-4c04-a7a6-fd6e05f68171.png)

A visualization of the crosswalk average image


Figure 5. Traffic Light Average Image

![image](https://user-images.githubusercontent.com/47839751/221709158-b052cc6d-0a3e-4f95-bb65-ca4272a0f9bf.png)

A visualization of the traffic light average image


2.1.3 Image Histograms

![image](https://user-images.githubusercontent.com/47839751/221709235-e03f517a-28b7-408d-ad52-fb510b1c5e52.png)

![image](https://user-images.githubusercontent.com/47839751/221709292-e09923c3-d777-47fb-9078-2e35ba6495d0.png)


We also used colored histograms to visualize the distribution of pixel values across the color channels (R, G, B) for each class of images in the dataset.The x-axis of each histogram represents the pixel value range, which is divided into a fixed number of bins; 256 in this case. The y-axis represents the frequency of pixels in each bin, normalized by the total number of pixels in the image.
By examining the histograms, we can get a sense of the color composition of each class of images. For example, if a class of images has a large peak in the red channel and a smaller peak in the blue channel, we can infer that the images in that class tend to have a warm color tone with more red than blue. If a class of images has roughly equal frequency across all color channels, we can infer that the images in that class have a balanced color composition. If a class of images has very low frequency in one or more color channels, we can infer that the images in that class are dominated by certain colors and lack diversity in color composition.
For the ‘Crosswalk’ class, the colors have relatively equal frequency throughout the chart, so it can be inferred that the color distribution in the class is balanced. There is a slightly higher distribution of blue in most of these images, but overall it is relatively evenly balanced. For the ‘Speed Limit’ class, there is slightly more distribution of green and blue colors than the red colors, but overall the color distribution is balanced. For the ‘Stop Sign’ class, there is a large amount of variance for the amount of blue coloring in the image. Some of the images are completely devoid of the blue color, while around 2.5% of the images classified as stop signs are almost completely saturated with blue. For the ‘Traffic Light’ class, the color distribution of the images are balanced. However, there are a few images completely devoid of one of the colors.
The color variations in the ‘Crosswalk” and ‘Speedlimit” show that the model can learn from those colors to classify between two classes. However, this is not the case for ‘Trafficlight’ and ‘Stop’, where the color variations are similar following almost a unimodal distribution for colors. Besides, there are some peaks in most of the classes around the sides of the images. These can be possible outliers that need to be handled by cropping the images around those peaks. For instance, ‘Stop’ can be cropped between 25 and 225 to eliminate the peaks at both ends. 


2.2 Outliers

We applied the Local Outlier Factor (LOF) on our images to detect images that are outliers compared to their nearest 10 neighbors. To be able to apply this predictor, we dropped the “class” and “filename” columns, so the model would not be skewed towards the majority class in the dataset. We deemed images with a score less than 0.97 as outliers. 

2.3 Image transformation

For our images with the bounding-box, we are additionally transforming our those generated images via a slight rotation. Doing this to the images makes sure that the bounding-box will still be around the class even if the image is at a different location.

2.4 Data Augmentation

Additional data must be created to balance the underrepresented classes.

