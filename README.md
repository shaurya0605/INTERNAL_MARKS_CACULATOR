# INTERNAL_MARKS_CACULATOR

# Student Marks Predictor

This is a simple program that helps you guess a student's final score based on their assignments, quizzes, attendance, and project scores.

# How to Use:
1. Open your terminal or command prompt.
2. Run the program by typing: python student-marks-simple/app/cli.py
3. You will see a prompt where you can type commands.

# Commands:
- train : Train the model using data from data/student_marks.csv
- predict : Enter your scores and get the guessed final score
- exit : To close the program

# Input Scores for Predicting:
- assignments_avg: Score out of 10
- quiz_avg: Score out of 10
- attendance_percent: Enter attendance percent (0 to 100). The program changes it internally as:
  - Less than 75% = 0 points
  - 75% to 80% = 1 point
  - 80% to 85% = 2 points
  - 85% to 90% = 3 points
  - 90% to 95% = 4 points
  - 95% to 100% = 5 points
- project_score: Score out of 20

# Output:
- The guessed final score is out of 45.

# Data File:
The data/student_marks.csv file should have these columns:
assignments_avg, quiz_avg, attendance_percent, project_score, final_score

# Notes:
- Run 'train' before 'predict'
- Model saves weights in model_weights.txt after training
- No extra software needed, simple Python code

This is a student project to learn basic machine learning with Python.
