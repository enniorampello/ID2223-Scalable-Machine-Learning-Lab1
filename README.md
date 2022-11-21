# ID2223-Scalable-Machine-Learning-Lab1
Lab 1 of the course ID2223 Scalable Machine Learning at KTH.\
Group members: Ennio Rampello, Vishal Nedungadi

The objective of this work is the implementation of a complete Machine Learning pipeline for both training and inference. The pipeline consists of different components with different purposes, where each one can be scaled independently of the others, based on the specific load.
The following is a description of all the components and procedures that we have implemented in order to carry out the lab work.

   1. **Feature pipeline to create a Feature Group in Hopsworks.** This component allows us to store the Titanic dataset in a convenient manner in Hopsworks.
   2. **Training pipeline executed in Modal.** This component retrieves the Feature Group from Hopsworks and runs the training of a Logistic Regression classifier in Modal. The resulting trained model is then registered in Hopsworks.
   3. **UI for interactive inference using Gradio.** This component allows the end user to enter new passenger information by using a web UI, it infers whether the passeger would survive or not and displays the inferred result to the user.
   4. **Synthetic data generator.** Here we create new synthetic data starting from the type of passenger that we want to generate (survived or not survived) and the features are sampled from the distributions that we have identified in the dataset. We do not use all the features of the dataset, but we only restrict to: pclass, sex, age, sibsp, fare, parch, embarked. For each of the two classes of passengers, we have analysed the distributions of these features in order to have a more correct sampling when generating synthetic data.
   5. **Batch inference pipeline.** This component receives a new batch of synthetic data and infers the class of each sample.
   6. **UI to show historical performance.** This component shows a web UI displaying the most recent passenger prediction and outcom, as well as a confusion matrix with historical prediction performance.

### Repository Contents

   1. `hugging_face/` contains the code for inferencing on a new data point. [URL link](https://huggingface.co/spaces/vishalned/scalable_ml_lab1_0)
   2. `hugging_face_monitoring/` contains the code for displaying the results from the batch inferencing. [URL link](https://huggingface.co/spaces/vishalned/scalable_ml_lab1)
   3. `titanic/` contains all the rest of the pipeline files.
   
   
