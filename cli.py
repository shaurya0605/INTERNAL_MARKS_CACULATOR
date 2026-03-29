from model import train_from_csv, predict_single, WEIGHTS_FILE
import os

def main():
    print("Student Marks Predictor")
    print("Please train the `model` before making predictions.")
    print("Commands: train, predict, exit")

    while True:
        cmd = input(">>> ").strip().lower()

        if cmd == "exit":
            break

        elif cmd == "train":
            this_dir = os.path.dirname(os.path.abspath(__file__))
            csv_path = os.path.normpath(os.path.join(this_dir, "..", "data", "student_marks.csv"))
            mae, w = train_from_csv(csv_path)
            print("Training completed.")
            print("MAE:", round(mae, 2))
            print("Weights saved to:", WEIGHTS_FILE)

        elif cmd == "predict":
            if not os.path.exists(WEIGHTS_FILE):
                print("Model not trained yet. Run 'train' first.")
                continue

            try:
                a = float(input("assignments_avg (out of 10): "))
                q = float(input("quiz_avg (out of 10): "))
                att = float(input("attendance_percent (0-100): "))
                p = float(input("project_score (out of 20): "))
            except ValueError:
                print("Invalid input. Please enter numeric values.")
                continue

            if not (0 <= a <= 10):
                print("Assignments average must be between 0 and 10.")
                continue
            if not (0 <= q <= 10):
                print("Quiz average must be between 0 and 10.")
                continue
            if not (0 <= att <= 100):
                print("Attendance percent must be between 0 and 100.")
                continue
            if not (0 <= p <= 20):
                print("Project score must be between 0 and 20.")
                continue

            pred = predict_single(a, q, att, p)
            print("Predicted Final Score (out of 45):", round(pred, 2))

        else:
            print("Unknown command.")

if __name__ == "__main__":
    main()