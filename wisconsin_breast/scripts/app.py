from ml_utils.run_models import run


def main():
    run("../train.csv", "../val.csv")
    print("DONE")


main()
