from flask import Flask, render_template, redirect, request
from pull import pull_all

app = Flask(__name__, template_folder = "template")

@app.route("/")
def index():
   return render_template("index.html")

@app.route("/pull", methods=['GET', 'POST'])
def pull():
   
   # added this to grab the form Home and Away "fields"
   home =request.form['Home']
   away =request.form['Away']

   # if request.method == 'POST':
   #       home = id.home
   #       away = id.away
   
   home_df, home_name, away_df, away_name, results = pull_all(home, away)
   # results = "Test Outcome"
   
   # if request.method == 'POST':
   #POST stuff
   #home_df to a table under home box
   #away_df to a table under away box
   #result to where ever we want it
   
   # return redirect('/', code=302)
   
   return render_template('index.html',results = results, home_name=home_name, away_name=away_name)
   
if __name__ == "__main__":
    app.run()