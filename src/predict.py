import argparse

from model_usage import DEFAULT_INPUT, format_prediction, load_final_model, predict_final_grade


def create_parser():
    parser = argparse.ArgumentParser(
        description="Jednostavan CLI primer za koriscenje finalnog modela."
    )
    parser.add_argument("--G2", type=float, default=DEFAULT_INPUT["G2"])
    parser.add_argument("--absences", type=float, default=DEFAULT_INPUT["absences"])
    parser.add_argument("--G1", type=float, default=DEFAULT_INPUT["G1"])
    parser.add_argument("--age", type=float, default=DEFAULT_INPUT["age"])
    parser.add_argument("--reason", default=DEFAULT_INPUT["reason"])
    parser.add_argument("--freetime", type=float, default=DEFAULT_INPUT["freetime"])
    parser.add_argument("--health", type=float, default=DEFAULT_INPUT["health"])
    parser.add_argument("--Fedu", type=float, default=DEFAULT_INPUT["Fedu"])
    parser.add_argument("--school", default=DEFAULT_INPUT["school"])
    parser.add_argument("--failures", type=float, default=DEFAULT_INPUT["failures"])

    return parser


def main():
    parser = create_parser()
    args = parser.parse_args()
    values = vars(args)

    model_bundle = load_final_model()
    prediction, input_df = predict_final_grade(values, model_bundle)

    print("Finalni model:", model_bundle["model"])
    print("Scenario:", model_bundle["scenario"])
    print("Ulazni atributi:")
    print(input_df.to_string(index=False))
    print(f"Predvidjena zavrsna ocena G3: {format_prediction(prediction)}")


if __name__ == "__main__":
    main()
