# NHL Matchup Calculator

A machine learning model to predict National Hockey League game outcomes based upon current season stats compared with 10+ years of past seasons of team data.

#Link to interactive website


#Why would predicting NHL games be interesting you ask?
    The group of us first got together on the basis that we wanted to investigate data in the sports realm and as the 2022 NHL season is now underway, we decided that hockey would be our winner. Hockey is a sport filled with data from team overall stats to individual performance measures like actual time on the ice and plus-minus which measures the net outcome of a team's goal differential when a certain player is on the ice. We decided on building a Win/Loss predictor model as the group thought it would be a fun and informative new method to appreciate both the consistency and randomness of the sport.

#Dataset Description
    Our data can be categorized in two groups: *Live* data (current season stats) which is pulled from NHL.com's API and a Kaggle dataset which gives detailed game outcomes for the last 10+ seasons dating back to 2013. Now, these two sources of data did not align perfectly so we needed to choose certain columns/features to omit. For instance, we figured that keeping "Goals" in our dataset would be too strong of a predictor so that has been removed from the model. 



#Outline of our Project
1. Identifying our datasource
    - Link to Kaggle Dataset: ["NHL Game Data"](https://www.kaggle.com/datasets/martinellis/nhl-game-data)
    - Description: The data represents all the official metrics measured for each game in the NHL in the past 10 years. The dataset consists of 26000+ game outcomes with stats from each team separately, yielding over 50,000 rows of data.
    - Link to NHL API: ["NHL API"](https://statsapi.web.nhl.com/api/v1/teams/1/stats/)
    - Description: The NHL.com API gives live, up-to date statistics in a variety of categories such as Powerplay %, Face-off %, and Save % among others. This API matches our team IDs from the Kaggle set in order to start aligning the datapoints.
     
Selected Topic: 
    For our project we are taking a look at NHL performance data and looking to predict user-selected matchups between current NHL teams based on their team stats and game outcomes over the last several years.
    We chose this topic as we all love hockey and in today's game, there is nearly endless data on both sides which can tell a story of how the games will turn out.
    
Data sources: 
    For this project we were able to find a dataset from Kaggle which contains team statistics for NHL teams dating back to the 2013 season. This dataset will give us a solid base of current NHL performance. Roster information is not included. In addition to the team stats, for expected game outcomes, we are using NHL.com's API which gives us a breakdown of every game up until the present day.
    
Question to answer:
    Between two user-selected current NHL teams using nearly a decade of analytics, who will emerge victorious in an "on-paper" matchup based on their team statistics without any interference of NHL referees' involvement?
    
          
Communication Protocols:
    We have agreed to meet both during class hours with added time on the front end as well as on Wednesdays on an as-needed basis based on team member availability.
