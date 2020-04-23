# MLB_prediction_production
![Home_page](readme_images/Home_screen.png)


This repository contains the data collection scripts as well as app deployment to AWS EC2 instance. 

The model instance is available here: <http://http://3.20.226.194/>

The repository containing detailed initial data collection, data experimentation, and model design can be found here:
<https://github.com/asoane34/retrosheet_analysis>

Each day, data is collected and new picks are generated and displayed. As of now, the model is operating with a timedelta of 1 year - there is no baseball season due to COVID19 at the moment, so it is just resimulating last year's season (for production testing). The picks are displayed in the following format

![picks_page](readme_images/Picks_screen.png)

