from src.train import load_data, build_preprocessor
from src.train import train_and_log_all


def main():
    print("[INFO] Starting preprocessing...")
    X_train, X_test, y_train, y_test = load_data()
    preprocessor = build_preprocessor(X_train)
    print("[INFO] Ending preprocessing\n")

    print("[INFO] Starting model training and logging...")
    train_and_log_all(X_train, X_test, y_train, y_test, preprocessor)
    print("[INFO] Ending model training and logging\n")


if __name__ == "__main__":
    main()
