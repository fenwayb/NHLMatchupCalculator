# NHL Matchup Calculator

A machine learning model to predict National Hockey League game outcomes based upon current season stats compared with 10+ years of past seasons of team data.

#Link to interactive website


## Why would predicting NHL games be interesting you ask?
The group of us first got together on the basis that we wanted to investigate data in the sports realm and as the 2022 NHL season is now underway, we decided that hockey would be our winner. Hockey is a sport filled with data from team overall stats to individual performance measures like actual time on the ice and plus-minus which measures the net outcome of a team's goal differential when a certain player is on the ice. We decided on building a Win/Loss predictor model as the group thought it would be a fun and informative new method to appreciate both the consistency and randomness of the sport.

## Dataset Description
Our data can be categorized in two groups: *Live* data (current season stats) which is pulled from NHL.com's API and a Kaggle dataset which gives detailed game outcomes for the last 10+ seasons. Now, these two sources of data did not align perfectly so we needed to choose certain columns/features to omit. For instance, we figured that keeping "Goals" in our dataset would be too strong of a predictor so that has been removed from the model. 


|team_id|franchiseId|shortName|teamName|abbreviation|link|
|:---|:---|:---|:---|:---|:---|
|1|23|New Jersey|Devils|NJD|/api/v1/teams/1|
|4|16|Philadelphia|Flyers|PHI|/api/v1/teams/4|
|26|14|Los Angeles|Kings|LAK|/api/v1/teams/26|
|14|31|Tampa Bay|Lightning|TBL|/api/v1/teams/14|
|6|6|Boston|Bruins|BOS|/api/v1/teams/6|
|3|10|NY Rangers|Rangers|NYR|/api/v1/teams/3|
|5|17|Pittsburgh|Penguins|PIT|/api/v1/teams/5|
|17|12|Detroit|Red Wings|DET|/api/v1/teams/17|
|28|29|San Jose|Sharks|SJS|/api/v1/teams/28|
|18|34|Nashville|Predators|NSH|/api/v1/teams/18|
|23|20|Vancouver|Canucks|VAN|/api/v1/teams/23|
|16|11|Chicago|Blackhawks|CHI|/api/v1/teams/16|
|9|30|Ottawa|Senators|OTT|/api/v1/teams/9|
|8|1|Montreal|Canadiens|MTL|/api/v1/teams/8|
|30|37|Minnesota|Wild|MIN|/api/v1/teams/30|
|15|24|Washington|Capitals|WSH|/api/v1/teams/15|
|19|18|St Louis|Blues|STL|/api/v1/teams/19|
|24|32|Anaheim|Ducks|ANA|/api/v1/teams/24|
|27|28|Phoenix|Coyotes|PHX|/api/v1/teams/27|
|2|22|NY Islanders|Islanders|NYI|/api/v1/teams/2|
|10|5|Toronto|Maple Leafs|TOR|/api/v1/teams/10|
|13|33|Florida|Panthers|FLA|/api/v1/teams/13|
|7|19|Buffalo|Sabres|BUF|/api/v1/teams/7|
|20|21|Calgary|Flames|CGY|/api/v1/teams/20|
|21|27|Colorado|Avalanche|COL|/api/v1/teams/21|
|25|15|Dallas|Stars|DAL|/api/v1/teams/25|
|29|36|Columbus|Blue Jackets|CBJ|/api/v1/teams/29|
|52|35|Winnipeg|Jets|WPG|/api/v1/teams/52|
|22|25|Edmonton|Oilers|EDM|/api/v1/teams/22|
|54|38|Vegas|Golden Knights|VGK|/api/v1/teams/54|
|12|26|Carolina|Hurricanes|CAR|/api/v1/teams/12|
|53|28|Arizona|Coyotes|ARI|/api/v1/teams/53|
|11|35|Atlanta|Thrashers|ATL|/api/v1/teams/11|     

Game and team statistics used to develop our machine learning model:
* **Home or Away**: Home ice advantage is factor of course.
* **Shooting Percentages**: Gotta put the biscuit on the basket.
* **PowerPlay Opportunities->Goals/Kills**: How often can they light the lamp with a man advantage and defend!
* **Save Percentages** Keepers gotta have some mitts.


## Outline of the Project
1. Identifying our datasource
    - Link to Kaggle Dataset: ["NHL Game Data"](https://www.kaggle.com/datasets/martinellis/nhl-game-data)
    - Description: The data represents all the official metrics measured for each game in the NHL in the past 10 years. The dataset consists of 26000+ game outcomes with stats from each team separately, yielding over 50,000 rows of data.
    - Link to NHL API: ["NHL API"](https://statsapi.web.nhl.com/api/v1/teams/1/stats/)
    - Description: The NHL.com API gives live, up-to date statistics in a variety of categories such as Powerplay %, Face-off %, and Save % among others. This API matches our team IDs from the Kaggle set in order to start aligning our data.
    
2. Selected Topic: 
    -  For our project we are taking a look at NHL performance data and looking to predict user-selected matchups between current NHL teams based on their team stats and game outcomes over the last several years.
    -  We chose this topic as we all love hockey and in today's game, there is nearly endless data on both sides which can tell a story of how the games will turn out.

3. Question to answer:
    - Between two user-selected current NHL teams using nearly a decade of analytics, who will emerge victorious in an "on-paper" matchup based on their team statistics without any interference of NHL referees' involvement?
4. Dataset Limitations:
    - Our game outcomes data has just 10 seasons of data whereas the NHL has been around since the early 20th century. One caveat in our favor is that the game has changed heavily since its first season over 100 years ago with the pace of play and overall team strategy.
5. Recommendations for further analysis:
    - More complex analysis can definitely be developed from our model. We can add a number of statistics to dive deeper such as figuring out the goals.

    
## Data sources: 
For this project we were able to find a dataset from Kaggle which contains team statistics for NHL teams dating back to the 2013 season. This dataset will give us a solid base of current NHL performance. Roster information is not included. In addition to the team stats, for expected game outcomes, we are using NHL.com's API which gives us a breakdown of every game up until the present day.
    

    

Communication Protocols:
    We have agreed to meet both during class hours with added time on the front end as well as on Wednesdays on an as-needed basis based on team member availability.
