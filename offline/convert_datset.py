import glob
import json


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
                lstm_dataset[key] = converted_input
            if key in output_keys:
                converted_output = convert_episode_output(value, input_steps, output_steps)
                lstm_dataset["out_" + key] = converted_output
    save_json(out_path=out_path, data=lstm_dataset)
