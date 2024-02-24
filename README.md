# Soccer-Event-Detection
An event is a specific occurrence of something that happens in a certain time and a certain place involving one or more participants, 
which can frequently be described as a change of state. Event detection is an important step in extracting knowledge from the video. 
The goal of event detection is to identify event instance(s) in data and, if existing, 
identify the event type as well as all of its participants and attributes.

The goal of the project is to correctly classify test images: the model should be able 
to discern between soccer-related events and general images, and also classify the specific event occurring.

In this paper, we propose a deep learning approach to detect events in a soccer match emphasizing 
the distinction between images of red and yellow cards and the correct detection of the images 
of selected events from other images. This method includes the following three modules: 
i) the variational autoencoder (VAE) module to differentiate between soccer images and others image, 
ii) the image classification module to classify the images ofevents,
iii) the fine-grain image classification module to classify the images of red and yellow cards. 

# Dataset
Soccer Event Dataset. The dataset contains images taken from UCL and EL football matches regarding seven main events: corner kick, penalty, free kick, red card, yellow card, tackle, substitution. [Dataset](https://github.com/FootballAnalysis/footballanalysis/tree/main/Dataset/Soccer%20Event%20Dataset%20(Image))

# References
Karimi, A., Toosi, R., & Akhaee, M. A. (2021). Soccer event detection using deep learning. arXiv preprint arXiv:2102.04331.
