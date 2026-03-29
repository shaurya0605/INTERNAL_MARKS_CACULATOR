import csv
import os

WEIGHTS_FILE = "model_weights.txt"

def attendance_to_points(attendance_percent):
    if attendance_percent < 75:
        return 0
    elif attendance_percent < 80:
        return 1
    elif attendance_percent < 85:
        return 2
    elif attendance_percent < 90:
        return 3
    elif attendance_percent < 95:
        return 4
    else:
        return 5

def load_dataset(csv_path):
    """
    Loads student_marks.csv and returns:
    X = [[assignments_avg, quiz_avg, attendance_points, project_score], ...]
    y = [final_score, final_score, ...]
    """

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    X = []
    y = []

    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        required_cols = [
            "assignments_avg",
            "quiz_avg",
            "attendance_percent",
            "project_score",
            "final_score"
        ]

        for col in required_cols:
            if col not in reader.fieldnames:
                raise ValueError(f"Missing column '{col}' in CSV file.")

        for row in reader:
            a = float(row["assignments_avg"])  # out of 10
            q = float(row["quiz_avg"])         # out of 10
            att_percent = float(row["attendance_percent"])  # out of 100%
            att_points = attendance_to_points(att_percent)
            p = float(row["project_score"])    # out of 20
            fs = float(row["final_score"])     # raw final score

            X.append([a, q, att_points, p])
            y.append(fs)

    return X, y

def train_from_csv(csv_path):
    """
    Trains a simple linear regression model on the CSV dataset using pure Python.
    Saves weights to a file.
    Returns mean absolute error and weights.
    """
    X, y = load_dataset(csv_path)

    # Add bias term to X
    X_bias = [[1] + features for features in X]

    # Calculate weights using normal equation without numpy
    # w = (X^T X)^-1 X^T y
    # Implement matrix operations manually

    # Helper functions for matrix operations
    def transpose(matrix):
        return list(map(list, zip(*matrix)))

    def matmul(A, B):
        result = []
        for i in range(len(A)):
            row = []
            for j in range(len(B[0])):
                s = 0
                for k in range(len(B)):
                    s += A[i][k] * B[k][j]
                row.append(s)
            result.append(row)
        return result

    def matinv(matrix):
        n = len(matrix)
        AM = [row[:] for row in matrix]
        I = [[float(i == j) for i in range(n)] for j in range(n)]

        for fd in range(n):
            if AM[fd][fd] == 0:
                for i in range(fd+1, n):
                    if AM[i][fd] != 0:
                        AM[fd], AM[i] = AM[i], AM[fd]
                        I[fd], I[i] = I[i], I[fd]
                        break
                else:
                    raise ValueError("Matrix is singular and cannot be inverted")
            fdScaler = 1.0 / AM[fd][fd]
            for j in range(n):
                AM[fd][j] *= fdScaler
                I[fd][j] *= fdScaler
            for i in range(n):
                if i == fd:
                    continue
                crScaler = AM[i][fd]
                for j in range(n):
                    AM[i][j] -= crScaler * AM[fd][j]
                    I[i][j] -= crScaler * I[fd][j]
        return I

    XT = transpose(X_bias)
    XTX = matmul(XT, X_bias)
    XTX_inv = matinv(XTX)
    XTy = matmul(XT, [[val] for val in y])
    w_matrix = matmul(XTX_inv, XTy)
    w = [row[0] for row in w_matrix]

    with open(WEIGHTS_FILE, "w") as f:
        f.write(" ".join(map(str, w)))

    preds = []
    for features in X_bias:
        pred = sum(w[i]*features[i] for i in range(len(w)))
        preds.append(pred)
    mae = sum(abs(preds[i] - y[i]) for i in range(len(y))) / len(y)

    return mae, w

def predict_single(assignments_avg, quiz_avg, attendance_percent, project_score):
    """
    Predicts final score out of 45 using saved weights and input features.

    Input:
    - assignments_avg (out of 10)
    - quiz_avg (out of 10)
    - attendance_percent (out of 100%)
    - project_score (out of 20)

    Attendance percent is converted to points internally.
    The predicted final score is scaled to be out of 45.
    """
    if not os.path.exists(WEIGHTS_FILE):
        raise FileNotFoundError(f"Model weights file '{WEIGHTS_FILE}' not found. Train the model first.")

    with open(WEIGHTS_FILE, "r") as f:
        w = list(map(float, f.read().strip().split()))

    att_points = attendance_to_points(attendance_percent)

    x = [1, assignments_avg, quiz_avg, att_points, project_score]
    pred_raw = sum(w[i]*x[i] for i in range(len(w)))

    max_assignments = 10
    max_quiz = 10
    max_attendance = 5
    max_project = 20

    max_raw = w[0] + w[1]*max_assignments + w[2]*max_quiz + w[3]*max_attendance + w[4]*max_project
    if max_raw != 0:
        pred_scaled = (pred_raw / max_raw) * 45
    else:
        pred_scaled = pred_raw

    return pred_scaled 