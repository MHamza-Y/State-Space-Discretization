import glob
import json
import os

import h5py
import numpy as np


def convert_dict_list_to_numpy(data):
    for key in data:
        data[key] = np.array(data[key])


def load_rllib_dataset(glob_regex):
    json_files = glob.glob(glob_regex)
    data = []
    for file in json_files:
        with open(file) as f:
            for line in f:
                episode_obj = json.loads(line)
                convert_dict_list_to_numpy(episode_obj)
                data.append(episode_obj)

    return data


def save_json(out_path, data):
    with open(out_path, 'a', encoding="utf8") as f:
        # save htfi
        json.dump(data, f)


def merge_rllib_out(rllib_files):
    data = load_rllib_dataset(rllib_files)
    mod_data = {}
    for obj in data:
        for key, value in obj.items():
            if key not in "type":
                if key in mod_data:
                    mod_data[key] = np.append(mod_data[key], value, axis=0)
                else:
                    mod_data[key] = value
    return mod_data


def merge_rllib_out_filtered(rllib_files, filter_fn=None):
    data = load_rllib_dataset(rllib_files)
    mod_data = {}
    for obj in data:
        if filter_fn:
            if not filter_fn(obj):
                continue
        for key, value in obj.items():
            if key not in "type":
                if key in mod_data:
                    mod_data[key] = np.append(mod_data[key], value, axis=0)
                else:
                    mod_data[key] = value
    return mod_data


def covert_episode_input(input_array, input_steps, output_steps):
    converted_input = []
    for count, value in enumerate(input_array):
        if count < len(input_array) - input_steps - output_steps:
            input_sample = input_array[count:count + input_steps]
            converted_input.append(input_sample)

    return np.asarray(converted_input)


def convert_episode_output(output_array, input_steps, output_steps):
    converted_output = []
    for count, value in enumerate(output_array):
        if input_steps <= count < len(output_array) - output_steps:
            output_sample = output_array[count:count + output_steps]
            converted_output.append(output_sample)

    return np.asarray(converted_output)


def merge_multiple_features(data, features: list, out_key_name, out_dict):
    if len(features) > 1:

        merged_data = []
        for feature in features:
            merged_data.append(data[feature])

        merged_data = np.dstack(merged_data)
    else:
        merged_data = data[features[0]]

    if out_key_name in out_dict:
        out_dict[out_key_name] = np.append(out_dict[out_key_name], merged_data, axis=0)
    else:

        out_dict[out_key_name] = merged_data


def save_hdf(out_path, data, h5_key, write_mode='w'):
    # df = pd.DataFrame(data)
    # df.to_hdf(out_path, h5_key)
    hf = h5py.File(out_path, write_mode)
    grp = hf.create_group(h5_key)
    for key in data:
        grp.create_dataset(key, data=data)
    hf.close()


def save_numpy(out_path, data):
    np.save(out_path, data)


def create_lstm_dataset(rllib_files, out_path, input_steps, output_steps, h5_key, input_keys=None, output_keys=None, ):
    if os.path.exists(out_path):
        os.remove(out_path)
    if output_keys is None:
        output_keys = ["obs", "actions"]

    if input_keys is None:
        input_keys = ["obs", "actions"]

    data = load_rllib_dataset(rllib_files)
    print(len(data))
    merged_data = {}
    for ep_id, obj in enumerate(data):
        print(f"Episode: {ep_id}")
        lstm_episode_input_dataset = {}
        lstm_episode_output_dataset = {}
        for key, value in obj.items():
            if key in input_keys:
                converted_input = covert_episode_input(value, input_steps, output_steps)

                lstm_episode_input_dataset[key] = converted_input

            if key in output_keys:
                converted_output = convert_episode_output(value, input_steps, output_steps)

                lstm_episode_output_dataset[key] = converted_output

        merge_multiple_features(lstm_episode_input_dataset, input_keys, "merged_input", merged_data)
        merge_multiple_features(lstm_episode_output_dataset, output_keys, "merged_output", merged_data)

    for key in merged_data:
        print(np.shape(merged_data[key]))
    # save_json(out_path=out_path, data=merged_data)
    save_numpy(out_path=out_path, data=merged_data)


def covert_episode_input_lookahead(input_array, input_steps, output_steps):
    converted_input = []
    for count, value in enumerate(input_array):
        if count < len(input_array) - input_steps - output_steps:
            input_sample = input_array[count:count + input_steps + output_steps - 1]
            converted_input.append(input_sample)

    return np.asarray(converted_input)


def create_look_ahead_lstm_dataset(rllib_files, out_path, input_steps, output_steps, input_keys=None, output_keys=None):
    if os.path.exists(out_path):
        os.remove(out_path)
    if output_keys is None:
        output_keys = ["obs"]

    if input_keys is None:
        input_keys = ["obs", "actions"]

    data = load_rllib_dataset(rllib_files)

    merged_data = {}
    for ep_id, obj in enumerate(data):
        print(f"Episode: {ep_id}")
        lstm_episode_input_dataset = {}
        lstm_episode_output_dataset = {}
        for key, value in obj.items():

            if key in input_keys:
                converted_input = covert_episode_input_lookahead(value, input_steps, output_steps)

                lstm_episode_input_dataset[key] = converted_input

            if key in output_keys:
                converted_output = convert_episode_output(value, input_steps, output_steps)

                lstm_episode_output_dataset[key] = converted_output

        merge_multiple_features(lstm_episode_input_dataset, input_keys, "merged_input", merged_data)
        merge_multiple_features(lstm_episode_output_dataset, output_keys, "merged_output", merged_data)

    for key in merged_data:
        print(key)
        print(np.shape(merged_data[key]))
    # save_json(out_path=out_path, data=merged_data)
    save_numpy(out_path=out_path, data=merged_data)
