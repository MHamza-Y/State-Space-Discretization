import glob
import json

import numpy as np
import pandas as pd


def load_rllib_dataset(glob_regex):
    json_files = glob.glob(glob_regex)
    data = []
    for file in json_files:
        with open(file) as f:
            for line in f:
                episode_obj = json.loads(line)
                data.append(episode_obj)

    return data


def save_json(out_path, data):
    with open(out_path, 'w') as f:
        json.dump(data, f)


def merge_rllib_out(rllib_files, out_path):
    data = load_rllib_dataset(rllib_files)
    mod_data = {}
    for obj in data:
        for key, value in obj.items():
            if key not in "type":
                if key in mod_data:
                    mod_data[key].extend(value)
                else:
                    mod_data[key] = value

    save_json(out_path, mod_data)


def covert_episode_input(input_array, input_steps, output_steps):
    converted_input = []
    for count, value in enumerate(input_array):
        if count < len(input_array) - input_steps - output_steps:
            input_sample = input_array[count:count + input_steps]
            input_sample.reverse()
            converted_input.append(input_sample)

    return converted_input


def convert_episode_output(output_array, input_steps, output_steps):
    converted_output = []
    for count, value in enumerate(output_array):
        if input_steps <= count < len(output_array) - output_steps:
            converted_output.append(output_array[count:count + output_steps])

    return converted_output


def merge_row(row):
    print(row.shape)
    return np.hstack(row)


def merge_multiple_features(data, features: list, out_key_name):
    if len(features) > 1:
        df = pd.DataFrame(data)

        converted_rows = df[features].to_numpy(dtype=object)
        # print(np.hstack(converted_rows[1]))
        # converted_rows = numpy.transpose(converted_rows)
        # print(converted_rows[0][1])
        # print(converted_rows[1][0])
        vf = np.vectorize(merge_row, signature="(n,m)->(k)")
        # vf(converted_rows)
        # merged_array = []
        # for i in data_rows:
        #
        #     row = [data[feature][i] for feature in features]
        #     row = np.hstack(row)
        #     if i is 0:
        #         merged_array = np.asarray([row])
        #     else:
        #         merged_array = np.append(merged_array, [row], axis=0)
        # data[out_key_name] = merged_array.tolist()

        # data[out_key_name] = np.hstack(converted_rows).tolist()
        # print(vf(converted_rows))
        # data[out_key_name] = vf(converted_rows).tolist()
        data[out_key_name] = [np.hstack(converted_rows[i]).tolist() for i in range(converted_rows.shape[0])]


def create_lstm_dataset(rllib_files, out_path, input_steps, output_steps, input_keys=None, output_keys=None):
    if output_keys is None:
        output_keys = ["obs", "actions"]

    if input_keys is None:
        input_keys = ["obs", "actions"]

    data = load_rllib_dataset(rllib_files)
    lstm_dataset = {}
    for obj in data:

        for key, value in obj.items():
            if key in input_keys:
                converted_input = covert_episode_input(value, input_steps, output_steps)
                if key in lstm_dataset:
                    lstm_dataset[key].extend(converted_input)
                else:
                    lstm_dataset[key] = converted_input

            if key in output_keys:
                converted_output = convert_episode_output(value, input_steps, output_steps)
                key = "out_" + key
                if key in lstm_dataset:
                    lstm_dataset[key].extend(converted_output)
                else:
                    lstm_dataset[key] = converted_output

    merge_multiple_features(lstm_dataset, input_keys, "merged_input")
    merge_multiple_features(lstm_dataset, output_keys, "merged_output")

    save_json(out_path=out_path, data=lstm_dataset)
