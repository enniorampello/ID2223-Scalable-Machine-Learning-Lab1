import os
import modal
    
BACKFILL=False
LOCAL=True

if LOCAL == False:
   stub = modal.Stub()
   image = modal.Image.debian_slim().pip_install(["hopsworks","joblib","seaborn","sklearn","dataframe-image"]) 

   @stub.function(image=image, schedule=modal.Period(days=1), secret=modal.Secret.from_name("scalable_ml"))
   def f():
       g()


# sample from bernoulli distribution
def bernoulli(sample_prob):
    import random
    return 1 if random.uniform(0,1) < sample_prob else 0

def generate_passenger(survived, age_min, age_max, sex_prob, fare_min, fare_max):
    """
    Returns a DataFrame containing one random iris flower
    """
    import pandas as pd
    import random
    import numpy as np


    df = pd.DataFrame(
        {
            "Pclass": [random.choice([1,2,3])],
            "Sex": [int(bernoulli(sex_prob))],
            "Age": [random.uniform(age_min, age_max)],
            "SibSp": [random.choice([0,1,2,3,4,5])],
            "Parch": [random.choice([0,1,2,3,4,5,6])],
            "Fare": [random.uniform(fare_min, fare_max)],
            "Embarked": [int(random.choice([0,1,2]))]
        }
    )
    df["Survived"] = survived

    return df


def get_random_passenger():
    """
    Returns a DataFrame containing one random iris flower
    """
    import pandas as pd
    import random

    survived_df = generate_passenger(1, 0.42, 80, 0.32, 0, 512.33)
    not_survived_df = generate_passenger(0, 1, 74, 0.85, 0, 263)
    
    # randomly pick one of these 2 and write it to the featurestore
    if random.choice([0,1]) == 0:
        print("survived df")
        df = survived_df 
    else: 
        print("not survived df")
        df = not_survived_df
    
    return df
    


def g():
    import hopsworks
    import pandas as pd

    project = hopsworks.login(api_key_value="otd1BvtKwvlF8OC1.Y8Kyt1QpZqDPMRNPIF3KvVGuFJpRdxIy39879ueQwymTgSDUU9vWKFMOnBqsyxfk")
    fs = project.get_feature_store()

    if BACKFILL == True:
        titanic_df = pd.read_csv("https://raw.githubusercontent.com/ID2223KTH/id2223kth.github.io/master/assignments/lab1/titanic.csv")
    else:
        titanic_df = get_random_passenger()
        # for i in range(0, 9):
        #     titanic_df = titanic_df.append(get_random_passenger())

    
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