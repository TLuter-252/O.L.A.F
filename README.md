O.L.A.F.
Outlier & Low-frequency Analysis Framework

What this project does
I use AIS ship tracking data to:

Run a hypothesis test to check if Class A and Class B vessels behave differently in “low traffic” areas.
Build a regression model (logistic regression) to predict which vessels look like outliers.
This is useful for military application because it can help point analysts to activity whose movement patterns are unusual, so they can spend time on the most important tracks first.

Data I need to add
I put my CSV here:

data/Florida_Routes.csv

My notebook expects these columns:

MMSI, VesselName, BaseDateTime, LAT, LON, SOG, COG, Status, TransceiverClass
Files in my repo
scripts/H Test and Log Reg.ipynb This is my capstone notebook. It includes the hypothesis test and the regression model.

scripts/READY_TO_BRIEF_APP_v3.py This is my Streamlit app file....it takes a wicked long time to run.

data/ I put Florida_Routes.csv here.

Proposal
Question
Do Class A and Class B vessels have different “low traffic percent” behavior in the Florida area?

MVP (Minimum Viable Product)
Load and filter AIS data for the Florida area
Create a baseline traffic grid
Score each vessel with low_traffic_percent
Ran a hypothesis test (Welch’s t-test)
MVP+ (after MVP is working)
I build a logistic regression model to predict “outlier” vessels
Hypothesis test
Null hypothesis (H0): Class A and Class B have the same average low_traffic_percent.

Alternate hypothesis (H1): Class A and Class B have different average low_traffic_percent.

Test used: Welch’s t-test Reason: I have two groups (A vs B) and I do not assume equal variance.

Significance level (alpha): 0.05 (chosen before I run the test)

My notebook prints:

sample sizes for each group
t-statistic and p-value
decision (reject or fail to reject H0)
Regression model
Model type: Logistic Regression (classification)

Goal: I predict is_outlier based on vessel-level features (speed, course, ranges, time span, etc.)

How I make sure it is properly fit:

I use a pipeline (scaling + logistic regression)
I use 5-fold Stratified Cross Validation
I evaluate with ROC AUC and Average Precision (PR AUC)
Class imbalance:

I handle it with class_weight="balanced"
Decision threshold:

I choose it using the precision-recall curve to meet a recall target
How to run
I put Florida_Routes.csv into data/
I open and run the notebook: scripts/01_hypothesis_test_and_regression.ipynb
If you have made it this far, thank you for the class Chad. This was awesome and I’ve learned so much.

PS... it needs more pie charts.
