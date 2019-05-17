# Jigsaw Unintended Bias in Toxicity Classification

The objective of this project is to detect toxic comments and minimize unintended model bias.
## Steps to run the project
- Open AWS S3 console and upload sparkproject_2.11-0.1.jar to S3 bucket.
- Add train data from jigsaw to S3Bucket
- Open AWS EMR console select a cluster and start the application form the add step.
- Run the class Project and pass the arguments path to the training data and path to the output.
- Once the job is done we can see the result in S3 as Output/project folder

We can see the test results of the toxicity analysis of the given data with different Estimation metrics

Logistic Regression metrics are as below:
Logistic Regression accuracy is: 	0.9197021533507748
Logistic Regression precision is: 	0.845852050878052
Logistic Regression Recall is: 	0.9197021533507748
Logistic Regression F1Score is: 	0.8812325905887495

# Declaration
This project was done as a part of Bigdata Course at UTD
