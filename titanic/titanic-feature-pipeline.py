import os
import modal
    
LOCAL=True

if LOCAL == False:
   stub = modal.Stub()
   image = modal.Image.debian_slim().pip_install(["hopsworks","joblib","seaborn","sklearn","dataframe-image"]) 

   @stub.function(image=image, schedule=modal.Period(days=1), secret=modal.Secret.from_name("scalable_ml"))
   def f():
       g()

def g():
    import hopsworks
    import pandas as pd

    project = hopsworks.login(api_key_value="otd1BvtKwvlF8OC1.Y8Kyt1QpZqDPMRNPIF3KvVGuFJpRdxIy39879ueQwymTgSDUU9vWKFMOnBqsyxfk")
    fs = project.get_feature_store()
    titanic_df = pd.read_csv("https://raw.githubusercontent.com/ID2223KTH/id2223kth.github.io/master/assignments/lab1/titanic.csv")

    # feature engineering of the titanic dataset
    cols = ['Name', 'Ticket', 'Cabin', 'PassengerId']
    titanic_df = titanic_df.drop(cols, axis=1)
    titanic_df = titanic_df.dropna()
    titanic_df['Sex'] = titanic_df['Sex'].map( {'female': 0, 'male': 1} )
    titanic_df['Embarked'] = titanic_df['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} )

    print(titanic_df.head())
    
    titanic_fg = fs.get_or_create_feature_group(
        name="titanic_modal_1",
        version=1,
        primary_key=["Pclass","Sex","Age","SibSp","Parch","Fare","Embarked"], 
        description="Titanic dataset")
    titanic_fg.insert(titanic_df, write_options={"wait_for_job" : False})

if __name__ == "__main__":
    if LOCAL == True :
        g()
    else:
        with stub.run():
            f()