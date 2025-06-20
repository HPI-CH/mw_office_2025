import copy
import pandas as pd
from datetime import datetime, timedelta


class Array_Size_Exception(Exception):
    """Raised when both arrays have more than one item"""
    pass
class Unequal_array_size_Exception(Exception):
    """Raised when both arrays have unequal size"""
    pass

class Other_Than_Two_Synchronization_Tappings_Occurence(Exception):
    """Raised when the number of synchronization routines read in a log file is unequal to two occurences"""
    pass


def get_timestamp_start_of_experiment_from_log(log_file):
    START_KEY = 'Press [SPACE] to start. started at '
    start_time = None
    for line in log_file:
        if START_KEY in line:
            start_time = line.split(START_KEY)[1]
            print('Lab Experiment was started at %s' % start_time)
            break
    return start_time


def get_synchronization_timestamps_from_log(log_file, check_for_correct_tappings_number=True):
    synchronization_counter = 0
    start_spacebar_taps_logged = False
    copy_for_start_spacebar_tapping = False
    copy_for_end_spacebar_tapping = False
    start_spacebar_taps_logs = []
    start_spacebar_taps_times = []
    end_spacebar_taps_logs = []
    end_spacebar_taps_times = []

    for line in log_file:
        if "Routine Muse - Psychopy Synchronization" in line:
            synchronization_counter += 1
            if not copy_for_start_spacebar_tapping and not start_spacebar_taps_logged:
                copy_for_start_spacebar_tapping = True
            elif copy_for_start_spacebar_tapping:
                copy_for_start_spacebar_tapping = False
                start_spacebar_taps_logged = True
            elif start_spacebar_taps_logged and not copy_for_end_spacebar_tapping:
                copy_for_end_spacebar_tapping = True
            elif copy_for_end_spacebar_tapping:
                copy_for_end_spacebar_tapping = False
            continue
        if copy_for_start_spacebar_tapping:
            start_spacebar_taps_logs.append(line)
        if copy_for_end_spacebar_tapping:
            end_spacebar_taps_logs.append(line)

    for elem in start_spacebar_taps_logs:
        if 'New trial' in elem:
            elem_info = elem.split(' \tEXP \tNew trial (rep=')
            start_spacebar_taps_times.append(timedelta(seconds=float(elem_info[0])))

    for elem in end_spacebar_taps_logs:
        if 'New trial' in elem:
            elem_info = elem.split(' \tEXP \tNew trial (rep=')
            end_spacebar_taps_times.append(timedelta(seconds=float(elem_info[0])))

    if check_for_correct_tappings_number and synchronization_counter != 4:     # There's more or less than two syncs
        print('NUMBER OF TAPS (%d ; COUNTING START AND END OF SYNC-SEQUENCE) IS NOT TWO' % synchronization_counter)
        raise Other_Than_Two_Synchronization_Tappings_Occurence

    num_taps = 4
    dictionary = {}

    times_of_taps_dicts = [start_spacebar_taps_times, end_spacebar_taps_times]

    for i in range(int(synchronization_counter / 2)):
        dictionary["synchronization_trial_" + str(i)] = {}
        dictionary["synchronization_trial_" + str(i)]['sync_global_start_timestamp'] = times_of_taps_dicts[i][0]
        dictionary["synchronization_trial_" + str(i)]['sync_global_end_timestamp'] = times_of_taps_dicts[i][-1]

        space_presses_array = []
        for j in range(num_taps):
            space_presses_array.append(times_of_taps_dicts[i][j])

        dictionary["synchronization_trial_" + str(i)]['space_presses'] = space_presses_array

    return dictionary


def get_panas_results_from_log(log_file):
    START_OF_PANAS_TRIAL_IN_LOG = " \tEXP \tRoutine PANAS started at "
    END_OF_PANAS_TRIAL_IN_LOG = " \tEXP \tRoutine PANAS finished at "

    # STEP ZERO: Define dictionaries for both the pairs of PANAS, as we ran in issues using the same dictionary
    first_individual_panas_values = {
        "sliderInterested: markerPos = ": None,
        "sliderDistressed: markerPos = ": None,
        "sliderExcited: markerPos = ": None,
        "sliderUpset: markerPos = ": None,
        "sliderStrong: markerPos = ": None,
        "sliderGuilty: markerPos = ": None,
        "sliderScared: markerPos = ": None,
        "sliderHostile: markerPos = ": None,
        "sliderEnthusiastic: markerPos = ": None,
        "sliderProud: markerPos = ": None,
        "sliderIrritable: markerPos = ": None,
        "sliderAlert: markerPos = ": None,
        "sliderAshamed: markerPos = ": None,
        "sliderInspired: markerPos = ": None,
        "sliderNervous: markerPos = ": None,
        "sliderDetermined: markerPos = ": None,
        "sliderAttentive: markerPos = ": None,
        "sliderJittery: markerPos = ": None,
        "sliderActive: markerPos = ": None,
        "sliderAfraid: markerPos = ": None
    }

    # Sadly, there's no way of determining when, e.g. textInterested, started, other than via using the hard-coded
    # order of elements and using the Delay-Routine timestamps
    first_individual_panas_slides_times = {
        START_OF_PANAS_TRIAL_IN_LOG: None,
        END_OF_PANAS_TRIAL_IN_LOG: None,
        "textInterested.started": None,
        "textGuilty.started": None,
        "textIrritable.started": None,
        "textDetermined.started": None
    }

    second_individual_panas_values = {
        "sliderInterested: markerPos = ": None,
        "sliderDistressed: markerPos = ": None,
        "sliderExcited: markerPos = ": None,
        "sliderUpset: markerPos = ": None,
        "sliderStrong: markerPos = ": None,
        "sliderGuilty: markerPos = ": None,
        "sliderScared: markerPos = ": None,
        "sliderHostile: markerPos = ": None,
        "sliderEnthusiastic: markerPos = ": None,
        "sliderProud: markerPos = ": None,
        "sliderIrritable: markerPos = ": None,
        "sliderAlert: markerPos = ": None,
        "sliderAshamed: markerPos = ": None,
        "sliderInspired: markerPos = ": None,
        "sliderNervous: markerPos = ": None,
        "sliderDetermined: markerPos = ": None,
        "sliderAttentive: markerPos = ": None,
        "sliderJittery: markerPos = ": None,
        "sliderActive: markerPos = ": None,
        "sliderAfraid: markerPos = ": None
    }

    second_individual_panas_slides_times = {
        START_OF_PANAS_TRIAL_IN_LOG: None,
        END_OF_PANAS_TRIAL_IN_LOG: None,
        "textInterested.started": None,
        "textGuilty.started": None,
        "textIrritable.started": None,
        "textDetermined.started": None
    }

    PANAS_SLIDES_IDS = ["textInterested.started", "textGuilty.started", "textIrritable.started",
                        "textDetermined.started"]

    # First, iterate through the whole file and extract values for each answer
    search_this_line = False
    # structure per trial: dict::individual_panas_values, dict::individual_panas_slides_times
    panas_trials = []
    panas_slide_counter = 0
    slide_change_event_marker = " \tEXP \tRoutine Delay started at "
    current_panas_trial = None
    current_panas_trial_times = None
    first_panas_trial_done = False

    for line in log_file:
        if START_OF_PANAS_TRIAL_IN_LOG in line:
            # print("TRIAL FOUND")
            search_this_line = True
            panas_slide_counter = 0
            if not first_panas_trial_done:
                current_panas_trial = first_individual_panas_values
                current_panas_trial_times = first_individual_panas_slides_times
            else:
                current_panas_trial = second_individual_panas_values
                current_panas_trial_times = second_individual_panas_slides_times
            current_panas_trial_times[START_OF_PANAS_TRIAL_IN_LOG] = timedelta(seconds=float(line.split(" ")[0]))
            print("TRIAL_START_TIME: %s" % timedelta(seconds=float(line.split(" ")[0])))
        elif END_OF_PANAS_TRIAL_IN_LOG in line:
            # print("TRIAL END FOUND")
            search_this_line = False
            first_panas_trial_done = True
            current_panas_trial_times[END_OF_PANAS_TRIAL_IN_LOG] = timedelta(seconds=float(line.split(" ")[0]))
            panas_trials.append([current_panas_trial, current_panas_trial_times])
            print("TRIAL_END_TIME: %s" % timedelta(seconds=float(line.split(" ")[0])))

        if search_this_line:
            if slide_change_event_marker in line:
                current_panas_trial_times[PANAS_SLIDES_IDS[panas_slide_counter]] = timedelta(
                    seconds=float(line.split(" ")[0])
                )
                panas_slide_counter += 1
                continue    # skip searching the dict for this line, as there can not be any more useful information
            for key in current_panas_trial.keys():
                if key in line:
                    split_result = line.split(key)[1].split("\n")[0]
                    current_panas_trial[key] = float(split_result) if split_result != 'None' else None

    # SECOND, CONTINUE WITH THIS INFORMATION AND STORE PER PANAS_TRIAL THE RESULTS PROPERLY
    dictionary = {}
    for i, trial in enumerate(panas_trials):
        dictionary["panas_trial_" + str(i)] = {}
        dictionary["panas_trial_" + str(i)]['PANAS_global_start_timestamp'] = trial[1][START_OF_PANAS_TRIAL_IN_LOG]
        dictionary["panas_trial_" + str(i)]['PANAS_global_end_timestamp'] = trial[1][END_OF_PANAS_TRIAL_IN_LOG]
        dictionary["panas_trial_" + str(i)]['textInterested.started'] = trial[1]["textInterested.started"]
        dictionary["panas_trial_" + str(i)]['sliderInterested.response'] = trial[0]['sliderInterested: markerPos = ']
        dictionary["panas_trial_" + str(i)]['sliderDistressed.response'] = trial[0]['sliderDistressed: markerPos = ']
        dictionary["panas_trial_" + str(i)]['sliderExcited.response'] = trial[0]['sliderExcited: markerPos = ']
        dictionary["panas_trial_" + str(i)]['sliderUpset.response'] = trial[0]['sliderUpset: markerPos = ']
        dictionary["panas_trial_" + str(i)]['sliderStrong.response'] = trial[0]['sliderStrong: markerPos = ']
        dictionary["panas_trial_" + str(i)]['textGuilty.started'] = trial[1]["textGuilty.started"]
        dictionary["panas_trial_" + str(i)]['sliderGuilty.response'] = trial[0]['sliderGuilty: markerPos = ']
        dictionary["panas_trial_" + str(i)]['sliderScared.response'] = trial[0]['sliderScared: markerPos = ']
        dictionary["panas_trial_" + str(i)]['sliderHostile.response'] = trial[0]['sliderHostile: markerPos = ']
        dictionary["panas_trial_" + str(i)]['sliderEnthusiastic.response'] = trial[0]['sliderEnthusiastic: markerPos = ']
        dictionary["panas_trial_" + str(i)]['sliderProud.response'] = trial[0]['sliderProud: markerPos = ']
        dictionary["panas_trial_" + str(i)]['textIrritable.started'] = trial[1]["textIrritable.started"]
        dictionary["panas_trial_" + str(i)]['sliderIrritable.response'] = trial[0]['sliderIrritable: markerPos = ']
        dictionary["panas_trial_" + str(i)]['sliderAlert.response'] = trial[0]['sliderAlert: markerPos = ']
        dictionary["panas_trial_" + str(i)]['sliderAshamed.response'] = trial[0]['sliderAshamed: markerPos = ']
        dictionary["panas_trial_" + str(i)]['sliderInspired.response'] = trial[0]['sliderInspired: markerPos = ']
        dictionary["panas_trial_" + str(i)]['sliderNervous.response'] = trial[0]['sliderNervous: markerPos = ']
        dictionary["panas_trial_" + str(i)]['textDetermined.started'] = trial[1]["textDetermined.started"]
        dictionary["panas_trial_" + str(i)]['sliderDetermined.response'] = trial[0]['sliderDetermined: markerPos = ']
        dictionary["panas_trial_" + str(i)]['sliderAttentive.response'] = trial[0]['sliderAttentive: markerPos = ']
        dictionary["panas_trial_" + str(i)]['sliderJittery.response'] = trial[0]['sliderJittery: markerPos = ']
        dictionary["panas_trial_" + str(i)]['sliderActive.response'] = trial[0]['sliderActive: markerPos = ']
        dictionary["panas_trial_" + str(i)]['sliderAfraid.response'] = trial[0]['sliderAfraid: markerPos = ']

    return dictionary


def get_nasa_results_from_log(log_file):
    Q1_MEANING = "Mental Demand"
    Q2_MEANING = "Physical Demand"
    Q3_MEANING = "Temporal Demand"
    Q4_MEANING = "Performance"
    Q5_MEANING = "Effort"
    Q6_MEANING = "Frustration"

    Q1_RESPONSE = "mental_demand_slider.response"
    Q2_RESPONSE = "physical_demand_slider.response"
    Q3_RESPONSE = "temporal_demand_slider.response"
    Q4_RESPONSE = "performance_slider.response"
    Q5_RESPONSE = "effort_slider.response"
    Q6_RESPONSE = "frustration_demand_slider.response"

    PAIR_23_32 = "Physical Demand vs Temporal Demand"
    KEY_23_32 = "physical_demand__vs__temporal_demand"
    PAIR_45_54 = "Performance vs Effort"
    KEY_45_54 = "performance__vs__effort"
    PAIR_12_21 = "Mental Demand vs Physical Demand"
    KEY_12_21 = "mental_demand__vs__physical_demand"
    PAIR_56_65 = "Effort vs Frustration"
    KEY_56_65 = "effort__vs__frustration"
    PAIR_24_42 = "Physical Demand vs Performance"
    KEY_24_42 = "physical_demand__vs__performance"
    PAIR_15_51 = "Mental Demand vs Effort"
    KEY_15_51 = "mental_demand__vs__effort"
    PAIR_46_64 = "Performance vs Frustration"
    KEY_46_64 = "performance_vs__frustration"
    PAIR_25_52 = "Physical Demand vs Effort"
    KEY_25_52 = "physical_demand__vs__effort"
    PAIR_14_41 = "Mental Demand vs Performance"
    KEY_14_41 = "mental_demand__vs__performance"
    PAIR_34_43 = "Temporal Demand vs Performance"
    KEY_34_43 = "temporal_demand__vs__performance"
    PAIR_26_62 = "Physical Demand vs Frustration"
    KEY_26_62 = "physical_demand__vs__frustration"
    PAIR_35_53 = "Temporal Demand vs Effort"
    KEY_35_53 = "temporal_demand__vs__effort"
    PAIR_16_61 = "Mental Demand vs Frustration"
    KEY_16_61 = "mental_demand__vs__frustration"
    PAIR_13_31 = "Mental Demand vs Temporal Demand"
    KEY_13_31 = "mental_demand__vs__temporal_demand"
    PAIR_36_63 = "Temporal Demand vs Frustration"
    KEY_36_63 = "temporal_demand__vs__frustration"

    ANSWER_FIRST = "first"
    ANSWER_SECOND = "second"

    # First, iterate through the whole file and identify lower and upper bounds for each answer, as well as threshold
    import re
    import numpy as np

    regex_pattern = r"pos=\([0-9][0-9][0-9],[0-9][0-9][0-9]\)"
    pattern = re.compile(regex_pattern, re.IGNORECASE)

    TEMP_START_KEY = "Routine NASA started"
    TEMP_END_KEY = "Routine Nasa finished"
    search_this_line = False

    pos_keys = []
    nasa_trials_log_lines = []
    current_nasa_trial = []

    for line in log_file:
        if TEMP_START_KEY in line:
            search_this_line = True
            current_nasa_trial = []
        elif TEMP_END_KEY in line:
            search_this_line = False
            current_nasa_trial.append(line)
            nasa_trials_log_lines.append(current_nasa_trial)

        if search_this_line:
            current_nasa_trial.append(line)
            temp_result = re.findall(pattern, line)
            if len(temp_result) > 0:
                pos_keys.append(temp_result[0])
            if len(temp_result) > 1:
                print("ERROR! INVESTIGATE HERE!")
                print(line)
                break

    x_pos = []
    y_pos = []

    for key in pos_keys:
        split_key = key.split(",")
        x_pos.append(int(split_key[0][-3:]))
        y_pos.append(int(split_key[1][:3]))

    np_x_pos = np.asarray(x_pos)
    np_y_pos = np.asarray(y_pos)

    min_y_pos = np_y_pos.min()
    max_y_pos = np_y_pos.max()
    min_x_pos = np_x_pos.min()
    max_x_pos = np_x_pos.max()

    threshold_y = min_y_pos + ((max_y_pos - min_y_pos) / 2)
    threshold_x = min_x_pos + ((max_x_pos - min_x_pos) / 2)

    pairwise_nasa_dict = {
        "Question32": "PAIR_23_32",
        "Question54": "PAIR_45_54",
        "Question21": "PAIR_12_21",
        "Question65": "PAIR_56_65",
        "Question42": "PAIR_24_42",
        "Question51": "PAIR_15_51",
        "Question64": "PAIR_46_64",
        "Question52": "PAIR_25_52",
        "Question41": "PAIR_14_41",
        "Question43": "PAIR_34_43",
        "Question62": "PAIR_26_62",
        "Question53": "PAIR_35_53",
        "Question61": "PAIR_16_61",
        "Question31": "PAIR_13_31",
        "Question63": "PAIR_36_63",
        "Question23": "PAIR_23_32",
        "Question45": "PAIR_45_54",
        "Question12": "PAIR_12_21",
        "Question56": "PAIR_56_65",
        "Question24": "PAIR_24_42",
        "Question15": "PAIR_15_51",
        "Question46": "PAIR_46_64",
        "Question25": "PAIR_25_52",
        "Question14": "PAIR_14_41",
        "Question34": "PAIR_34_43",
        "Question26": "PAIR_26_62",
        "Question35": "PAIR_35_53",
        "Question16": "PAIR_16_61",
        "Question13": "PAIR_13_31",
        "Question36": "PAIR_36_63"
    }

    # Second, get NASA results for each trial
    nasa_trials_results = []
    for nasa_trial in nasa_trials_log_lines:
        current_nasa_trial_individual_answers = None
        current_nasa_trial_pairwise_answers = None
        current_nasa_start_timestamp = timedelta(seconds=float(nasa_trial[0].split(" ")[0]))
        current_nasa_end_timestamp = timedelta(seconds=float(nasa_trial[-1].split(" ")[0]))

        # Take care of the individual trials
        trial_individual_nasa_answers = {
            "sliderQ1": -10,
            "sliderQ2": -10,
            "sliderQ3": -10,
            "sliderQ4": -10,
            "sliderQ5": -10,
            "sliderQ6": -10
        }

        # Take care of the pairwise trials
        trial_nasa_pos_keys = {
            "PAIR_23_32": None,
            "PAIR_45_54": None,
            "PAIR_12_21": None,
            "PAIR_56_65": None,
            "PAIR_24_42": None,
            "PAIR_15_51": None,
            "PAIR_46_64": None,
            "PAIR_25_52": None,
            "PAIR_14_41": None,
            "PAIR_34_43": None,
            "PAIR_26_62": None,
            "PAIR_35_53": None,
            "PAIR_16_61": None,
            "PAIR_13_31": None,
            "PAIR_36_63": None
        }

        END_KEY_PAIRWISE_TRIAL = "Routine Delay started at "

        current_nasa_pair_idx = None
        current_nasa_pair_pos_value = None
        for line in nasa_trial:

            # Check for the individual answers, first making sure we can parse this lines result to float
            for idx in trial_individual_nasa_answers.keys():
                if (idx in line and "Created" not in line and "autoLog" not in line and "styleTweaks" not in line
                        and "contrast" not in line and "autoDraw" not in line):
                    tmp_split_read = line.split("markerPos = ")[1]
                    if tmp_split_read is not None and "None" not in tmp_split_read:
                        trial_individual_nasa_answers[idx] = float(tmp_split_read)
                        continue

            # Check for the pairwise answers
            if (END_KEY_PAIRWISE_TRIAL in line
                    and current_nasa_pair_idx is not None
                    and current_nasa_pair_pos_value is not None):   # Save pos-data and reset
                pos_answer = ANSWER_FIRST if current_nasa_pair_pos_value[1] >= threshold_y else ANSWER_SECOND
                trial_nasa_pos_keys[current_nasa_pair_idx] = pos_answer     # Store if first or second was selected
                current_nasa_pair_idx = None
                current_nasa_pair_pos_value = None

            for idx in pairwise_nasa_dict.keys():
                if idx in line:
                    current_nasa_pair_idx = pairwise_nasa_dict[idx]     # Set correct identifier for pairwise question
                    continue

            temp_result = re.findall(pattern, line)
            if len(temp_result) > 0:
                split_key = temp_result[0].split(",")
                x_pos = int(split_key[0][-3:])
                y_pos = int(split_key[1][:3])
                current_nasa_pair_pos_value = [x_pos, y_pos]

        # KEEP THIS ONE LINE IN THE END, AS THE TRIAL IS NOT FINISHED WITH END_KEY_PAIRWISE_TRIAL
        pos_answer = ANSWER_FIRST if (current_nasa_pair_pos_value is not None) and (current_nasa_pair_pos_value[1] >= threshold_y) else ANSWER_SECOND
        trial_nasa_pos_keys[current_nasa_pair_idx] = pos_answer  # Store if first or second was selected
        current_nasa_trial_pairwise_answers = trial_nasa_pos_keys
        current_nasa_trial_individual_answers = trial_individual_nasa_answers
        nasa_trials_results.append([current_nasa_start_timestamp, current_nasa_end_timestamp,
                                    current_nasa_trial_individual_answers, current_nasa_trial_pairwise_answers])

    # Third, CONTINUE WITH THIS INFORMATION AND STORE PER NASA_TRIAL THE RESULTS PROPERLY
    dictionary = {}
    for i, trial in enumerate(nasa_trials_results):
        dictionary["nasa_trial_" + str(i)] = {}
        dictionary["nasa_trial_" + str(i)]['Nasa_global_start_timestamp'] = trial[0]    # timestamp in seconds
        dictionary["nasa_trial_" + str(i)]['Nasa_global_end_timestamp'] = trial[1]      # timestamp in seconds
        dictionary["nasa_trial_" + str(i)]['mental_demand_slider.response'] = trial[2]["sliderQ1"]
        dictionary["nasa_trial_" + str(i)]['physical_demand_slider.response'] = trial[2]["sliderQ2"]
        dictionary["nasa_trial_" + str(i)]['temporal_demand_slider.response'] = trial[2]["sliderQ3"]
        dictionary["nasa_trial_" + str(i)]['performance_slider.response'] = trial[2]["sliderQ4"]
        dictionary["nasa_trial_" + str(i)]['effort_slider.response'] = trial[2]["sliderQ5"]
        dictionary["nasa_trial_" + str(i)]['frustration_demand_slider.response'] = trial[2]["sliderQ6"]
        dictionary["nasa_trial_" + str(i)]['physical_demand__vs__temporal_demand'] = trial[3]["PAIR_23_32"]
        dictionary["nasa_trial_" + str(i)]['performance__vs__effort'] = trial[3]["PAIR_45_54"]
        dictionary["nasa_trial_" + str(i)]['mental_demand__vs__physical_demand'] = trial[3]["PAIR_12_21"]
        dictionary["nasa_trial_" + str(i)]['effort__vs__frustration'] = trial[3]["PAIR_56_65"]
        dictionary["nasa_trial_" + str(i)]['physical_demand__vs__performance'] = trial[3]["PAIR_24_42"]
        dictionary["nasa_trial_" + str(i)]['mental_demand__vs__effort'] = trial[3]["PAIR_15_51"]
        dictionary["nasa_trial_" + str(i)]['performance_vs__frustration'] = trial[3]["PAIR_46_64"]
        dictionary["nasa_trial_" + str(i)]['physical_demand__vs__effort'] = trial[3]["PAIR_25_52"]
        dictionary["nasa_trial_" + str(i)]['mental_demand__vs__performance'] = trial[3]["PAIR_14_41"]
        dictionary["nasa_trial_" + str(i)]['temporal_demand__vs__performance'] = trial[3]["PAIR_34_43"]
        dictionary["nasa_trial_" + str(i)]['physical_demand__vs__frustration'] = trial[3]["PAIR_26_62"]
        dictionary["nasa_trial_" + str(i)]['temporal_demand__vs__effort'] = trial[3]["PAIR_35_53"]
        dictionary["nasa_trial_" + str(i)]['mental_demand__vs__frustration'] = trial[3]["PAIR_16_61"]
        dictionary["nasa_trial_" + str(i)]['mental_demand__vs__temporal_demand'] = trial[3]["PAIR_13_31"]
        dictionary["nasa_trial_" + str(i)]['temporal_demand__vs__frustration'] = trial[3]["PAIR_36_63"]

    return dictionary


def get_affective_slider_results_from_log(log_file):
    # First, define all the marker keys to search for in the log file
    START_TIMESTAMP = " \tEXP \tRoutine Affective-Slider started at "
    END_TIMESTAMP = " \tEXP \tRoutine Affective-Slider finished at "
    SLIDER_VALENCE_STARTED = " \tEXP \tRoutine Affective-Slider (Routine) started at "
    SLIDER_VALENCE_RESPONSE = " \tEXP \tsliderValence: markerPos = "
    SLIDER_AROUSAL_RESPONSE = " \tEXP \tsliderArousal: markerPos = "
    affective_slider_answers = []
    current_affective_slider_trial = {}
    search_this_line = False

    # Second, iterate through the log file and store the values per affective slider trial
    for line in log_file:
        if START_TIMESTAMP in line:
            search_this_line = True
            current_affective_slider_trial = {}
            current_affective_slider_trial[START_TIMESTAMP] = timedelta(seconds=float(line.split(START_TIMESTAMP)[0]))
        elif END_TIMESTAMP in line:
            search_this_line = False
            current_affective_slider_trial[END_TIMESTAMP] = timedelta(seconds=float(line.split(END_TIMESTAMP)[0]))
            affective_slider_answers.append(current_affective_slider_trial)

        if search_this_line:
            if SLIDER_VALENCE_STARTED in line:
                current_affective_slider_trial[SLIDER_VALENCE_STARTED] = (
                    timedelta(seconds=float(line.split(SLIDER_VALENCE_STARTED)[0])))
            elif SLIDER_VALENCE_RESPONSE in line:
                valence_response = line.split(SLIDER_VALENCE_RESPONSE)[1].split("\n")[0]
                current_affective_slider_trial[SLIDER_VALENCE_RESPONSE] = (
                    float(valence_response)) if valence_response != "None" else None
            elif SLIDER_AROUSAL_RESPONSE in line:
                arousal_response = line.split(SLIDER_AROUSAL_RESPONSE)[1].split("\n")[0]
                current_affective_slider_trial[SLIDER_AROUSAL_RESPONSE] = (
                    float(arousal_response)) if arousal_response != "None" else None

    # Third, save each trials values in a proper dictionary
    dictionary = {}
    for i, trial in enumerate(affective_slider_answers):
        dictionary["affective_slider_trial_" + str(i)] = {}
        dictionary["affective_slider_trial_" + str(i)]['Affective_Slider_global_start_timestamp'] = trial[START_TIMESTAMP]
        dictionary["affective_slider_trial_" + str(i)]['Affective_Slider_global_end_timestamp'] = trial[END_TIMESTAMP]
        dictionary["affective_slider_trial_" + str(i)]['sliderValence.started'] = trial[SLIDER_VALENCE_STARTED]
        dictionary["affective_slider_trial_" + str(i)]['sliderValence.response'] = trial[SLIDER_VALENCE_RESPONSE]
        dictionary["affective_slider_trial_" + str(i)]['sliderArousal.response'] = trial[SLIDER_AROUSAL_RESPONSE]

    return dictionary


def get_likert_scale_results_from_log(log_file):
    # First, define all the marker keys to search for in the log file
    CHOICES = ['very, very low', 'neither low nor high', 'very, very high', 'low', 'high']
    START_TIMESTAMP = " \tEXP \tRoutine likert Scale started at "
    END_TIMESTAMP = " \tEXP \tRoutine likert Scale finished at "
    LIKERT_RESPONSE = " \tEXP \tLikert-Scale ("
    TYPE_MENTAL_EFFORT = "mental effort"
    TYPE_STRESS = "stress"
    KEY_MENTAL_EFFORT = "Likert_Scale_Rating_mental_effort"
    KEY_STRESS = "Likert_Scale_Rating_stress"
    likert_scale_answers = []
    current_likert_scale_trial = {}
    search_this_line = False

    # Second, iterate through the log file and store the values per affective slider trial
    for line in log_file:
        if START_TIMESTAMP in line:
            search_this_line = True
            current_likert_scale_trial = {}
            current_likert_scale_trial[START_TIMESTAMP] = timedelta(seconds=float(line.split(START_TIMESTAMP)[0]))
        elif END_TIMESTAMP in line:
            search_this_line = False
            current_likert_scale_trial[END_TIMESTAMP] = timedelta(seconds=float(line.split(END_TIMESTAMP)[0]))
            likert_scale_answers.append(current_likert_scale_trial)

        if search_this_line:
            if LIKERT_RESPONSE in line:
                response = line.split(LIKERT_RESPONSE)
                response_type = response[1].split(')')[0]
                current_likert_scale_trial["type"] = \
                    "mental_effort" if response_type == TYPE_MENTAL_EFFORT else TYPE_STRESS
                response_value = None
                if CHOICES[0] in line:
                    response_value = CHOICES[0]
                elif CHOICES[1] in line:
                    response_value = CHOICES[1]
                elif CHOICES[2] in line:
                    response_value = CHOICES[2]
                elif CHOICES[3] in line:
                    response_value = CHOICES[3]
                elif CHOICES[4] in line:
                    response_value = CHOICES[4]
                current_likert_scale_trial["response_value"] = response_value
                current_likert_scale_trial["value_key"] = (
                    KEY_MENTAL_EFFORT) if response_type == TYPE_MENTAL_EFFORT else KEY_STRESS

    # Third, save each trials values in a proper dictionary
    dictionary = {}
    for i, trial in enumerate(likert_scale_answers):
        dictionary["likert_trial_" + str(i)] = {}
        dictionary["likert_trial_" + str(i)]['likert_start_timestamp'] = trial[START_TIMESTAMP]
        dictionary["likert_trial_" + str(i)]['likert_end_timestamp'] = trial[END_TIMESTAMP]
        dictionary["likert_trial_" + str(i)]['likert_type'] = trial["type"]
        dictionary["likert_trial_" + str(i)][trial["value_key"]] = trial["response_value"]

    return dictionary


def get_relaxation_video_from_log(log_file):

    ROUTINE_START_MARKER = ' \tEXP \tRoutine Relaxation-Video started at'
    ROUTINE_END_MARKER = ' \tEXP \tRoutine Relaxation-Video finished at'

    start_ctr = 0
    end_ctr = 0

    start_time = 0
    end_time = 0

    dictionary = {}

    for line in log_file:
        if ROUTINE_START_MARKER in line:
            start_ctr += 1
            start_time = timedelta(seconds=float(line.split(ROUTINE_START_MARKER)[0]))
            continue
        if ROUTINE_END_MARKER in line:
            end_ctr += 1
            end_time = timedelta(seconds=float(line.split(ROUTINE_END_MARKER)[0]))
            continue

    if start_ctr == 1 and end_ctr == 1:
        dictionary['relaxation_video_start_timestamp'] = start_time
        dictionary['relaxation_video_stop_timestamp'] = end_time
    else:
        raise Array_Size_Exception
    return dictionary


def get_eye_closing_baseline_from_log(log_file):

    ROUTINE_START_MARKER = ' \tEXP \tRoutine Eye Closing Routine started at '
    ROUTINE_END_MARKER = ' \tEXP \tRoutine Eye Closing Routine finished at '

    start_ctr = 0
    end_ctr = 0

    start_time = 0
    end_time = 0

    dictionary = {}

    for line in log_file:
        if ROUTINE_START_MARKER in line:
            start_ctr += 1
            start_time = timedelta(seconds=float(line.split(ROUTINE_START_MARKER)[0]))
            continue
        if ROUTINE_END_MARKER in line:
            end_ctr += 1
            end_time = timedelta(seconds=float(line.split(ROUTINE_END_MARKER)[0]))
            continue

    if start_ctr == end_ctr:
        dictionary['eye_closing_start_timestamp'] = start_time
        dictionary['eye_closing_stop_timestamp'] = end_time
    else:
        raise Array_Size_Exception
    return dictionary


def get_arithmetic_results_from_log(log_file):
    # Sadly, we'll be unable to locate the trial-samples from the log-files, as these were not properly logged
    # However, we'll be able to recover all information for the actual trial

    EASY_KEYS = ["Arithmetix_Easy", "Arithmetix with NeededFiles/easy_tasks.csv started at ",
                 "Arithmetix with NeededFiles/easy_tasks.csv finished at "]
    HARD_KEYS = ["Arithmetix_Hard", "Arithmetix with NeededFiles/hard_tasks.csv started at ",
                 "Arithmetix with NeededFiles/hard_tasks.csv finished at "]

    NEW_TRIAL_KEY = ' \tEXP \tNew trial ('
    RESULT_KEY = ' \tDATA \tThe answer '
    TIME_FOR_RESULT_KEY = ' \tDATA \tThe time needed for this answer was '

    easy_start_time = 0
    easy_end_time = 0
    hard_start_time = 0
    hard_end_time = 0

    # will hold: a, op, b, cor. answer, part. answer, correct?, RT, Start-Time, End-Time
    current_trial_holder = ['a', 'op', 'b', 'answer', 'part_answer', 'correct?', 'RT', 'Start_Time', 'End_Time']
    easy_trials_info = []
    hard_trials_info = []

    easy_trial = False
    hard_trial = False

    for line in log_file:
        if EASY_KEYS[1] in line:
            easy_start_time = timedelta(seconds=float(line.split(' ')[0]))
            easy_trial = True
        elif EASY_KEYS[2] in line:
            easy_end_time = timedelta(seconds=float(line.split(' ')[0]))
            easy_trial = False
        elif HARD_KEYS[1] in line:
            hard_start_time = timedelta(seconds=float(line.split(' ')[0]))
            hard_trial = True
        elif HARD_KEYS[2] in line:
            hard_end_time = timedelta(seconds=float(line.split(' ')[0]))
            hard_trial = False
        if easy_trial or hard_trial:
            if NEW_TRIAL_KEY in line:
                trial_start_time = line.split(NEW_TRIAL_KEY)[0]
                trial_op_a = line.split("'a', ")[1].split(')')[0]
                trial_op_op = line.split("'op', ")[1].split(')')[0]
                trial_op_b = line.split("'b', ")[1].split(')')[0]
                trial_answer = line.split("'answer', ")[1].split(')')[0]
                current_trial_holder[0] = trial_op_a
                current_trial_holder[1] = trial_op_op
                current_trial_holder[2] = trial_op_b
                current_trial_holder[3] = trial_answer
                current_trial_holder[-2] = trial_start_time
                continue
            if RESULT_KEY in line:
                part_answer = line.split(RESULT_KEY)[1].split(' ')[0]
                part_answer_correct_bool = line.split(RESULT_KEY)[1].split('was ')[1].split(',')[0].strip()
                current_trial_holder[4] = part_answer
                current_trial_holder[5] = part_answer_correct_bool
                continue
            if TIME_FOR_RESULT_KEY in line:
                end_time = line.split(TIME_FOR_RESULT_KEY)[0]
                rt = line.split(TIME_FOR_RESULT_KEY)[1].strip()
                current_trial_holder[6] = rt
                current_trial_holder[-1] = end_time
                if easy_trial:
                    easy_trials_info.append(current_trial_holder.copy())
                elif hard_trial:
                    hard_trials_info.append(current_trial_holder.copy())
                continue

    dictionary = {}

    if easy_start_time != 0 and easy_end_time != 0 and hard_start_time != 0 and hard_end_time != 0:
        easy_trial_idx = 0 if easy_end_time.total_seconds() < hard_start_time.total_seconds() else 1
        hard_trial_idx = 1 if easy_trial_idx == 0 else 0
        log_results_to_store_in_dictionary = [
            [easy_start_time, easy_end_time, easy_trial_idx, easy_trials_info, "NeededFiles/easy_tasks.csv"],
            [hard_start_time, hard_end_time, hard_trial_idx, hard_trials_info, "NeededFiles/hard_tasks.csv"]
        ]
    elif easy_start_time == 0 or easy_end_time == 0:    # no easy arithmetic existing in logs
        print('ATTENTION! NOT USING THE STANDARD DATA EXTRACTION APPROACH! NO EASY ARITHMETIC!')
        log_results_to_store_in_dictionary = [
            [hard_start_time, hard_end_time, 0, hard_trials_info, "NeededFiles/hard_tasks.csv"]
        ]
    else:   # no hard arithmetic existing in logs
        print('ATTENTION! NOT USING THE STANDARD DATA EXTRACTION APPROACH! NO HARD ARITHMETIC!')
        log_results_to_store_in_dictionary = [
            [easy_start_time, easy_end_time, 0, easy_trials_info, "NeededFiles/easy_tasks.csv"]
        ]

    for idx, elem in enumerate(log_results_to_store_in_dictionary):     # For each info_block of [easy:hard]

        # check that initial timestamp is available and that next row has arithmetic value displayed
        dictionary['arithmetic_trial_' + str(elem[2])] = {}
        dictionary['arithmetic_trial_' + str(elem[2])]['arithmetic_trial_start_timestamp'] = elem[0]
        dictionary['arithmetic_trial_' + str(elem[2])]['arithmetic_trial_end_timestamp'] = elem[1]
        dictionary['arithmetic_trial_' + str(elem[2])]['csv_file_name'] = elem[4]
        dictionary['arithmetic_trial_' + str(elem[2])]['exercise_list'] = {}

        # current_trial_holder = ['a', 'op', 'b', 'answer', 'part_answer', 'correct?', 'RT', 'Start_Time', 'End_Time']
        for i, trial in enumerate(elem[3]):  # For each trial in trials_info
            trial_dict = {'variable_a': trial[0], 'operator': trial[1], 'variable_b': trial[2],
                          'correct_answer': trial[3], 'participant__answer': trial[4],
                          'participant_correct_answer_bool': trial[5], 'response_time': trial[6],
                          'text_display.started': trial[7]}
            dictionary['arithmetic_trial_' + str(elem[2])]['exercise_list']['exercise_' + str(i)] = trial_dict

    return dictionary


def get_n_back_from_log(log_file):
    # THE ISSUE HERE WILL BE THAT WE WILL NEED TO CHECK FOR EACH TRIAL IF THERE IS A DATA_RESPONSE_KEY PRESENT!
    # CHECKED IF THIS (TRUE) MEANS IT WAS EASY OR HARD N-BACK: TRUE MEANS EASY (1-Back) AND FALSE HARD (2-BACK)
    EASY_EXAMPLE_NBACK_ROUTINE_FINISHED = " \tEXP \tRoutine Example N-Back Trial (True) finished at "
    HARD_EXAMPLE_NBACK_ROUTINE_FINISHED = " \tEXP \tRoutine Example N-Back Trial (False) finished at "
    EXAMPLE_NBACK_ROUTINE_START_KEY = " \tEXP \tRoutine Instructions Example N-Back started at "
    EASY_NBACK_ROUTINE_STARTED = " \tEXP \tRoutine N-Back Trial (True) started at "
    HARD_NBACK_ROUTINE_STARTED = " \tEXP \tRoutine N-Back Trial (False) started at "
    EASY_NBACK_ROUTINE_FINISHED = " \tEXP \tRoutine N-Back Trial (True) finished at "
    HARD_NBACK_ROUTINE_FINISHED = " \tEXP \tRoutine N-Back Trial (False) finished at "
    NEW_TRIAL_KEY = ' \tEXP \tNew trial ('
    DATA_RESPONSE_KEY = " \tDATA \tKeypress: "

    easy_example_start_time = 0
    easy_example_end_time = 0
    easy_start_time = 0
    easy_end_time = 0
    hard_example_start_time = 0
    hard_example_end_time = 0
    hard_start_time = 0
    hard_end_time = 0
    temp_example_start_time = 0

    # will hold: participant_correct_answer_bool, block_color, correct_answer, participant_answer, response_time
    current_trial_holder = ['participant_correct_answer_bool', 'color', 'correct_answer', 'participant_answer', 'rt']
    current_trial_start_time = 0
    example_trials = []
    easy_examples = []
    hard_examples = []
    easy_trials_info = []
    hard_trials_info = []

    example_routine = False
    easy_routine = False
    hard_routine = False

    # Keep track of the elapsed lines to know if a (or especially no!) answer was given
    temp_trial_lines_counter = 0
    MAX_LOG_LINES_FOR_ONE_TRIAL_ANSWER = 16

    current_trial_holder = [None, None, None, None, None]  # Reset current_trial_holder
    for line in log_file:

        # Identify if any block has started/ended and set respective variables
        if EXAMPLE_NBACK_ROUTINE_START_KEY in line:
            temp_example_start_time = timedelta(seconds=float(line.split(' ')[0]))
            example_routine = True
            easy_routine = False
            hard_routine = False
        elif EASY_EXAMPLE_NBACK_ROUTINE_FINISHED in line:
            easy_example_start_time = temp_example_start_time
            easy_example_end_time = timedelta(seconds=float(line.split(' ')[0]))
            easy_examples = example_trials
            example_trials = []
            example_routine = False
            easy_routine = True
            hard_routine = False
        elif HARD_EXAMPLE_NBACK_ROUTINE_FINISHED in line:
            hard_example_start_time = temp_example_start_time
            hard_example_end_time = timedelta(seconds=float(line.split(' ')[0]))
            hard_examples = example_trials
            example_trials = []
            easy_routine = False
            example_routine = False
            hard_routine = True
        elif EASY_NBACK_ROUTINE_STARTED in line:
            easy_start_time = timedelta(seconds=float(line.split(' ')[0]))
            example_routine = False
            easy_routine = True
            hard_routine = False
        elif EASY_NBACK_ROUTINE_FINISHED in line:
            easy_end_time = timedelta(seconds=float(line.split(' ')[0]))
            example_routine = False
            easy_routine = False
            hard_routine = False
        elif HARD_NBACK_ROUTINE_STARTED in line:
            hard_start_time = timedelta(seconds=float(line.split(' ')[0]))
            example_routine = False
            easy_routine = False
            hard_routine = True
        elif HARD_NBACK_ROUTINE_FINISHED in line:
            hard_end_time = timedelta(seconds=float(line.split(' ')[0]))
            example_routine = False
            easy_routine = False
            hard_routine = False

        # Deal with the actual NBACK-data-holding lines of the log; As a reference here's the structure of trial_holder
        # current_trial_holder = ['participant_correct_answer_bool', 'color',
        #                         'correct_answer', 'participant_answer', 'rt']
        if example_routine or easy_routine or hard_routine:
            if NEW_TRIAL_KEY in line:
                # Unfortunately, we have to check here if a trial needs saving, as no log entry explicitly states this,
                # and we can only assume that by now all the required information has been extracted from the log file
                if current_trial_holder[0] is not None:
                    if example_routine:
                        example_trials.append(current_trial_holder)
                    elif easy_routine:
                        easy_trials_info.append(current_trial_holder)
                    elif hard_routine:
                        hard_trials_info.append(current_trial_holder)
                    current_trial_holder = [None, None, None, None, None]   # Reset current_trial_holder
                current_trial_start_time = timedelta(seconds=(float(line.split(NEW_TRIAL_KEY)[0])))
                trial_color = line.split("'colourtest': ")[1].split(',')[0]
                trial_correct_answer = line.split("'corresp': ")[1].split('}')[0]
                current_trial_holder[1] = trial_color
                current_trial_holder[2] = trial_correct_answer
                temp_trial_lines_counter = 0
            elif DATA_RESPONSE_KEY in line and current_trial_holder[1] is not None:     # ensure a valid trial started
                current_trial_response_time = (timedelta(seconds=(float(line.split(DATA_RESPONSE_KEY)[0])))
                                               - current_trial_start_time)
                current_trial_response = line.split(DATA_RESPONSE_KEY)[1].strip()
                current_trial_holder[0] = current_trial_response in current_trial_holder[2]
                current_trial_holder[3] = current_trial_response
                current_trial_holder[4] = current_trial_response_time
            if temp_trial_lines_counter >= MAX_LOG_LINES_FOR_ONE_TRIAL_ANSWER:  # Assume no answer was given for trial
                if current_trial_holder[4] is None:     # Check to see if really no answer was given, based on RT
                    current_trial_response_time = 'null'
                    current_trial_response = None
                    current_trial_holder[0] = current_trial_holder[2] == 'None' or current_trial_holder[2] is None
                    current_trial_holder[3] = current_trial_response
                    current_trial_holder[4] = current_trial_response_time
                    temp_trial_lines_counter = 0

            temp_trial_lines_counter += 1   # Keep track that we read one more line

    # Properly store the example trials in a dictionary
    example_dictionary = {}

    if easy_start_time != 0 and easy_end_time != 0 and hard_start_time != 0 and hard_end_time != 0:
        easy_example_trial_idx = 0 if easy_example_end_time.total_seconds() < hard_example_start_time.total_seconds() else 1
        hard_example_trial_idx = 1 if easy_example_end_time.total_seconds() < hard_example_start_time.total_seconds() else 0
        example_log_results_to_store_in_dictionary = [
            [easy_example_start_time, easy_example_end_time, easy_example_trial_idx, easy_examples, "Easy"],
            [hard_example_start_time, hard_example_end_time, hard_example_trial_idx, hard_examples, "Hard"]
        ]
    elif easy_start_time == 0 or easy_end_time == 0:    # no easy nback example existing in logs
        print('ATTENTION! NOT USING THE STANDARD DATA EXTRACTION APPROACH! NO EASY NBACK EXAMPLE!')
        example_log_results_to_store_in_dictionary = [
            [hard_example_start_time, hard_example_end_time, 0, hard_examples, "Hard"]
        ]
    else:   # no hard arithmetic existing in logs
        print('ATTENTION! NOT USING THE STANDARD DATA EXTRACTION APPROACH! NO HARD NBACK EXAMPLE!')
        example_log_results_to_store_in_dictionary = [
            [easy_example_start_time, easy_example_end_time, 0, easy_examples, "Easy"],
        ]

    # For each info_block of [easy:hard] examples
    for idx, elem in enumerate(example_log_results_to_store_in_dictionary):

        # check that initial timestamp is available and that next row has arithmetic value displayed
        example_dictionary['n_back_example_trial_' + str(elem[2])] = {}
        example_dictionary['n_back_example_trial_' + str(elem[2])]['n_back_example_trial_start_timestamp'] = elem[0]
        example_dictionary['n_back_example_trial_' + str(elem[2])]['n_back_example_trial_end_timestamp'] = elem[1]
        example_dictionary['n_back_example_trial_' + str(elem[2])]['n-back_type'] = elem[4]
        example_dictionary['n_back_example_trial_' + str(elem[2])]['exercise_list'] = {}

        # current_trial_holder = ['participant_correct_answer', 'color', 'correct_answer', 'participant_answer', 'rt']
        for i, trial in enumerate(elem[3]):  # For each trial in example_run
            trial_dict = {'participant_correct_answer_bool': trial[0], 'block_color': trial[1],
                          'correct_answer': trial[2], 'participant_answer': trial[3], 'response_time': trial[4]}
            example_dictionary['n_back_example_trial_' + str(elem[2])]['exercise_list']['exercise_' + str(i)] = trial_dict

    # Properly store the actual results in a dictionary
    actual_tasks_dictionary = {}

    if easy_start_time != 0 and easy_end_time != 0 and hard_start_time != 0 and hard_end_time != 0:
        easy_trial_idx = 0 if easy_end_time.total_seconds() < hard_start_time.total_seconds() else 1
        hard_trial_idx = 1 if easy_end_time.total_seconds() < hard_start_time.total_seconds() else 0
        trials_log_results_to_store_in_dictionary = [
            [easy_start_time, easy_end_time, easy_trial_idx, easy_trials_info, "Easy"],
            [hard_start_time, hard_end_time, hard_trial_idx, hard_trials_info, "Hard"]
        ]
    elif easy_start_time == 0 or easy_end_time == 0:    # no easy nback example existing in logs
        print('ATTENTION! NOT USING THE STANDARD DATA EXTRACTION APPROACH! NO EASY NBACK!')
        trials_log_results_to_store_in_dictionary = [
            [hard_start_time, hard_end_time, 0, hard_trials_info, "Hard"]
        ]
    else:   # no hard arithmetic existing in logs
        print('ATTENTION! NOT USING THE STANDARD DATA EXTRACTION APPROACH! NO HARD NBACK!')
        trials_log_results_to_store_in_dictionary = [
            [easy_start_time, easy_end_time, 0, easy_trials_info, "Easy"],
        ]

    # For each info_block of [easy:hard] examples
    for idx, elem in enumerate(trials_log_results_to_store_in_dictionary):

        # check that initial timestamp is available and that next row has arithmetic value displayed
        actual_tasks_dictionary['n_back_trial_' + str(elem[2])] = {}
        actual_tasks_dictionary['n_back_trial_' + str(elem[2])]['n_back_trial_start_timestamp'] = elem[0]
        actual_tasks_dictionary['n_back_trial_' + str(elem[2])]['n_back_trial_end_timestamp'] = elem[1]
        actual_tasks_dictionary['n_back_trial_' + str(elem[2])]['n-back_type'] = elem[4]
        actual_tasks_dictionary['n_back_trial_' + str(elem[2])]['exercise_list'] = {}

        # current_trial_holder = ['participant_correct_answer', 'color', 'correct_answer', 'participant_answer', 'rt']
        for i, trial in enumerate(elem[3]):  # For each trial in example_run
            trial_dict = {'participant_correct_answer_bool': trial[0], 'block_color': trial[1],
                          'correct_answer': trial[2], 'participant_answer': trial[3], 'response_time': trial[4]}
            actual_tasks_dictionary['n_back_trial_' + str(elem[2])]['exercise_list']['exercise_' + str(i)] = trial_dict

    return example_dictionary, actual_tasks_dictionary


def get_stroop_from_log(log_file):
    # EASY_EXAMPLE_STROOP_ROUTINE_FINISHED = " \tEXP \tRoutine Example Stroop level EASY finished at "
    HARD_EXAMPLE_STROOP_ROUTINE_FINISHED = " \tEXP \tRoutine Example Stroop level HARD finished at "
    # EASY_EXAMPLE_STROOP_ROUTINE_START_KEY = " \tEXP \tRoutine Example Stroop level EASY started at "
    HARD_EXAMPLE_STROOP_ROUTINE_START_KEY = " \tEXP \tRoutine Example Stroop level HARD started at "
    # EASY_STROOP_ROUTINE_STARTED = " \tEXP \tRoutine Stroop level EASY started at "
    HARD_STROOP_ROUTINE_STARTED = " \tEXP \tRoutine Stroop level HARD started at "
    # EASY_STROOP_ROUTINE_FINISHED = " \tEXP \tRoutine Stroop level EASY finished at "
    HARD_STROOP_ROUTINE_FINISHED = " \tEXP \tRoutine Stroop level HARD finished at "
    NEW_TRIAL_KEY = " \tEXP \tNew trial ("
    DATA_RESPONSE = " \tEXP \tParticipant answer: "
    INCORRECT_TERM = "is incorrect"
    RESPONSE_TIME_KEY = "Response time: "
    INCORRECT_RESPONSE_TIME_KEY = "Response time: []"

    easy_example_start_time = 0
    easy_example_end_time = 0
    easy_start_time = 0
    easy_end_time = 0
    hard_example_start_time = 0
    hard_example_end_time = 0
    hard_start_time = 0
    hard_end_time = 0

    # holds: participant_correct_answer_bool, word, colour, word_congruent_to_colour, participant_answer, response_time
    current_trial_holder = ["participant_correct_answer_bool", "word", "colour", "word_congruent_to_colour",
                            "participant_answer", "response_time"]
    current_trial_start_time = 0
    # easy_examples = []
    hard_examples = []
    # easy_trials_info = []
    hard_trials_info = []

    # easy_example_routine = False
    hard_example_routine = False
    # easy_routine = False
    hard_routine = False

    hard_stroop_routine_done = False

    current_trial_holder = [None, None, None, None, None, None]  # Reset current_trial_holder
    for line in log_file:

        # Identify if any block has started/ended and set respective variables
        # if EASY_EXAMPLE_STROOP_ROUTINE_START_KEY in line:
        #     easy_example_start_time = timedelta(seconds=float(line.split(' ')[0]))
        #     easy_example_routine = True
        #     hard_example_routine = False
        #     easy_routine = False
        #     hard_routine = False
        # elif EASY_EXAMPLE_STROOP_ROUTINE_FINISHED in line:
        #     easy_example_end_time = timedelta(seconds=float(line.split(' ')[0]))
        #     easy_example_routine = True
        #     hard_example_routine = False
        #     easy_routine = False
        #     hard_routine = False
        if HARD_EXAMPLE_STROOP_ROUTINE_START_KEY in line:
            hard_example_start_time = timedelta(seconds=float(line.split(' ')[0]))
            # easy_example_routine = False
            hard_example_routine = True
            # easy_routine = False
            hard_routine = False
        elif HARD_EXAMPLE_STROOP_ROUTINE_FINISHED in line:
            hard_example_end_time = timedelta(seconds=float(line.split(' ')[0]))
            # easy_example_routine = False
            hard_example_routine = True
            # easy_routine = False
            hard_routine = False
        # elif EASY_STROOP_ROUTINE_STARTED in line:
        #     easy_start_time = timedelta(seconds=float(line.split(' ')[0]))
        #     easy_example_routine = False
        #     hard_example_routine = False
        #     easy_routine = True
        #     hard_routine = False
        # elif EASY_STROOP_ROUTINE_FINISHED in line:
        #     easy_end_time = timedelta(seconds=float(line.split(' ')[0]))
        #     easy_example_routine = False
        #     hard_example_routine = False
        #     easy_routine = True
        #     hard_routine = False
        elif HARD_STROOP_ROUTINE_STARTED in line:
            hard_start_time = timedelta(seconds=float(line.split(' ')[0]))
            # easy_example_routine = False
            hard_example_routine = False
            # easy_routine = False
            hard_routine = True
        elif HARD_STROOP_ROUTINE_FINISHED in line:
            hard_end_time = timedelta(seconds=float(line.split(' ')[0]))
            # easy_example_routine = False
            hard_example_routine = False
            # easy_routine = False
            hard_routine = True
            hard_stroop_routine_done = True
            if hard_stroop_routine_done:
                print('Should now STOP trying to add more elements for STROOP HARD!!')
                break

        # Deal with the actual STROOP-data-holding lines of the log; As a reference here's the structure of trial_holder
        # current_trial_holder = ["participant_correct_answer_bool", "word", "colour", "word_congruent_to_colour",
        #                         "participant_answer", "response_time"]
        # if easy_example_routine or hard_example_routine or easy_routine or hard_routine:
        if hard_example_routine or hard_routine:
            if NEW_TRIAL_KEY in line and "'word':" in line:
                # New trial (rep=26, index=11): {'word': 'blue', 'colour': 'blue', 'congruent': 1, 'corrAns': 'b'}
                current_trial_start_time = line.split(NEW_TRIAL_KEY)[0]
                trial_word = line.split("'word': ")[1].split(',')[0].replace("'", "")
                trial_color = line.split("'colour': ")[1].split(',')[0].replace("'", "")
                trial_w_congruent_to_c = 'true' if int(line.split("'congruent': ")[1].split(',')[0]) == 1 else 'false'
                current_trial_holder[1] = trial_word
                current_trial_holder[2] = trial_color
                current_trial_holder[3] = trial_w_congruent_to_c
            elif DATA_RESPONSE in line:
                trial_participant_correct_answer_bool = None
                trial_participant_answer = None
                trial_response_time = None
                if INCORRECT_RESPONSE_TIME_KEY in line:     # participant answer was not given in time
                    trial_participant_correct_answer_bool = False
                    trial_participant_answer = 'None'
                    trial_response_time = "null"
                elif INCORRECT_TERM not in line:    # answer was given and correct
                    trial_participant_correct_answer_bool = True
                    trial_participant_answer = line.split(DATA_RESPONSE)[1].split("'")[0]
                    if '[' in line.split(RESPONSE_TIME_KEY)[1]:
                        trial_response_time = timedelta(seconds=float(line.split(RESPONSE_TIME_KEY)[1].replace('[', '').replace(']\n', '')))
                    else:
                        trial_response_time = timedelta(seconds=float(line.split(RESPONSE_TIME_KEY)[1]))
                else:   # answer was given but incorrect
                    trial_participant_correct_answer_bool = False
                    trial_participant_answer = line.split(DATA_RESPONSE)[1].split("'")[0]
                    if '[' in line.split(RESPONSE_TIME_KEY)[1]:
                        trial_response_time = timedelta(seconds=float(line.split(RESPONSE_TIME_KEY)[1].replace('[', '').replace(']\n', '')))
                    else:
                        trial_response_time = timedelta(seconds=float(line.split(RESPONSE_TIME_KEY)[1]))
                current_trial_holder[0] = trial_participant_correct_answer_bool
                current_trial_holder[4] = trial_participant_answer
                current_trial_holder[5] = trial_response_time

                # if easy_example_routine:
                #     easy_examples.append(current_trial_holder.copy())  
                #     print(current_trial_holder)
                #     current_trial_holder = [None, None, None, None, None, None]  # Reset current_trial_holder
                if hard_example_routine:
                    hard_examples.append(current_trial_holder.copy())  
                    # print(current_trial_holder)
                    current_trial_holder = [None, None, None, None, None, None]  # Reset current_trial_holder
                # elif easy_routine:
                #     easy_trials_info.append(current_trial_holder.copy())  
                #     print(current_trial_holder)
                #     current_trial_holder = [None, None, None, None, None, None]  # Reset current_trial_holder
                elif hard_routine:
                    hard_trials_info.append(current_trial_holder.copy())  
                    # print('Adding to hard_trials_info:')
                    # print(current_trial_holder)
                    current_trial_holder = [None, None, None, None, None, None]  # Reset current_trial_holder

    # Properly store the example trials in a dictionary
    example_dictionary = {}

    if type(easy_example_end_time) != type(42):
        easy_example_trial_idx = 0 if easy_example_end_time.total_seconds() < hard_example_start_time.total_seconds() else 1
        hard_example_trial_idx = 1 if easy_example_end_time.total_seconds() < hard_example_start_time.total_seconds() else 0
    else:
        hard_example_trial_idx = 0

    example_log_results_to_store_in_dictionary = [
        # [easy_example_start_time, easy_example_end_time, easy_example_trial_idx, easy_examples, "Easy"],
        [hard_example_start_time, hard_example_end_time, hard_example_trial_idx, hard_examples, "Hard"]
    ]

    # For each info_block of [easy:hard] examples
    for idx, elem in enumerate(example_log_results_to_store_in_dictionary):

        # check that initial timestamp is available and that next row has arithmetic value displayed
        example_dictionary['stroop_example_trial_' + str(elem[2])] = {}
        example_dictionary['stroop_example_trial_' + str(elem[2])]['stroop_example_trial_start_timestamp'] = elem[0]
        example_dictionary['stroop_example_trial_' + str(elem[2])]['stroop_example_trial_end_timestamp'] = elem[1]
        example_dictionary['stroop_example_trial_' + str(elem[2])]['stroop_type'] = elem[4]
        example_dictionary['stroop_example_trial_' + str(elem[2])]['exercise_list'] = {}

        # current_trial_holder = ["participant_correct_answer_bool", "word", "colour", "word_congruent_to_colour",
        #                         "participant_answer", "response_time"]
        for i, trial in enumerate(elem[3]):  # For each trial in example_run
            trial_dict = {'participant_correct_answer_bool': trial[0], 'word': trial[1], 'colour': trial[2],
                          'word_congruent_to_colour': trial[3], 'participant_answer': trial[4],
                          'response_time': trial[5]}
            example_dictionary['stroop_example_trial_' + str(elem[2])]['exercise_list']['exercise_' + str(i)] = trial_dict

    # Properly store the actual results in a dictionary
    actual_tasks_dictionary = {}

    if type(easy_end_time) != type(42):
        easy_trial_idx = 0 if easy_end_time.total_seconds() < hard_start_time.total_seconds() else 1
        hard_trial_idx = 1 if easy_end_time.total_seconds() < hard_start_time.total_seconds() else 0
    else: 
        hard_trial_idx = 0
    
    trials_log_results_to_store_in_dictionary = [
        # [easy_start_time, easy_end_time, easy_trial_idx, easy_trials_info, "Easy"],
        [hard_start_time, hard_end_time, hard_trial_idx, hard_trials_info, "Hard"]
    ]

    # For each info_block of [easy:hard] examples
    for idx, elem in enumerate(trials_log_results_to_store_in_dictionary):

        # check that initial timestamp is available and that next row has arithmetic value displayed
        actual_tasks_dictionary['stroop_trial_' + str(elem[2])] = {}
        actual_tasks_dictionary['stroop_trial_' + str(elem[2])]['stroop_trial_start_timestamp'] = elem[0]
        actual_tasks_dictionary['stroop_trial_' + str(elem[2])]['stroop_trial_end_timestamp'] = elem[1]
        actual_tasks_dictionary['stroop_trial_' + str(elem[2])]['stroop_type'] = elem[4]
        actual_tasks_dictionary['stroop_trial_' + str(elem[2])]['exercise_list'] = {}

        # current_trial_holder = ["participant_correct_answer_bool", "word", "colour", "word_congruent_to_colour",
        #                         "participant_answer", "response_time"]
        for i, trial in enumerate(elem[3]):  # For each trial in example_run
            trial_dict = {'participant_correct_answer_bool': trial[0], 'word': trial[1], 'colour': trial[2],
                          'word_congruent_to_colour': trial[3], 'participant_answer': trial[4],
                          'response_time': trial[5]}
            actual_tasks_dictionary['stroop_trial_' + str(elem[2])]['exercise_list']['exercise_' + str(i)] = trial_dict

    return example_dictionary, actual_tasks_dictionary


def get_sudoku_results_from_log(log_file):
    EASY_SUDOKU_ROUTINE_STARTED = " \tEXP \tRoutine Sudoku (True) started at "
    EASY_SUDOKU_ROUTINE_FINISHED = " \tEXP \tRoutine Sudoku (True) finished at "
    HARD_SUDOKU_ROUTINE_STARTED = " \tEXP \tRoutine Sudoku (False) started at "
    HARD_SUDOKU_ROUTINE_FINISHED = " \tEXP \tRoutine Sudoku (False) finished at "

    easy_start_time = 0
    easy_end_time = 0
    hard_start_time = 0
    hard_end_time = 0

    for line in log_file:

        # Identify if any block has started/ended and set respective variables
        if EASY_SUDOKU_ROUTINE_STARTED in line:
            easy_start_time = timedelta(seconds=float(line.split(' ')[0]))
        elif EASY_SUDOKU_ROUTINE_FINISHED in line:
            easy_end_time = timedelta(seconds=float(line.split(' ')[0]))
        elif HARD_SUDOKU_ROUTINE_STARTED in line:
            hard_start_time = timedelta(seconds=float(line.split(' ')[0]))
        elif HARD_SUDOKU_ROUTINE_FINISHED in line:
            hard_end_time = timedelta(seconds=float(line.split(' ')[0]))

    # Properly store the actual results in a dictionary
    actual_tasks_dictionary = {}

    if type(easy_end_time) != type(42):
        easy_trial_idx = 0 if easy_end_time.total_seconds() < hard_start_time.total_seconds() else 1
        hard_trial_idx = 1 if easy_end_time.total_seconds() < hard_start_time.total_seconds() else 0
    else:
        hard_trial_idx = 0

    trials_log_results_to_store_in_dictionary = [
        # [easy_start_time, easy_end_time, easy_trial_idx, "Easy"],
        [hard_start_time, hard_end_time, hard_trial_idx, "Hard"]
    ]

    # For each info_block of [easy:hard] examples
    for idx, elem in enumerate(trials_log_results_to_store_in_dictionary):

        # check that initial timestamp is available and that next row has arithmetic value displayed
        actual_tasks_dictionary['sudoku_trial_' + str(elem[2])] = {}
        actual_tasks_dictionary['sudoku_trial_' + str(elem[2])]['sudoku_trial_start_timestamp'] = elem[0]
        actual_tasks_dictionary['sudoku_trial_' + str(elem[2])]['sudoku_trial_end_timestamp'] = elem[1]
        actual_tasks_dictionary['sudoku_trial_' + str(elem[2])]['sudoku_type'] = elem[3]
        actual_tasks_dictionary['sudoku_trial_' + str(elem[2])]['sudoku_num_games_finished'] = 1.0

    return actual_tasks_dictionary


def parse_psychopy_log_file(log_path, check_for_correct_tappings_number=True):
    all_important_logs = []
    all_evaluations = []
    all_tasks = []

    tasks_dictionary = [
        # ["TASK", "START_STRING", "END_STRING"],
        ["Video", "Relaxation-Video started at ", "Relaxation-Video finished at "],
        ["Eye-Closing", "Eye-Closing started at ", "Eye-Closing finished at "],
        ["Arithmetix_Easy", "Arithmetix with NeededFiles/easy_tasks.csv started at ",
         "Arithmetix with NeededFiles/easy_tasks.csv finished at "],
        ["Arithmetix_Hard", "Arithmetix with NeededFiles/hard_tasks.csv started at ",
         "Arithmetix with NeededFiles/hard_tasks.csv finished at "],
        ["Sudoku_Easy", "Sudoku (False) started at ", "Sudoku (False) finished at "],
        ["Sudoku_Hard", "Sudoku (True) started at ", "Sudoku (True) finished at "],
        ["Stroop_Easy", "Stroop level EASY started at ", "Stroop level EASY finished at "],
        ["Stroop_Hard", "Stroop level HARD started at ", "Stroop level HARD finished at "],
        ["NBack_Easy", "Example N-Back Trial (False) started at ", "Global N-Back finished at "],
        ["NBack_Hard", "Example N-Back Trial (True) started at ", "Global N-Back finished at "],
    ]

    task_list = [
        "Routine Muse - Psychopy Synchronization",
        "Routine Relaxation-Video",
        "Routine Eye-Closing",
        "Routine Arithmetix",
        "Routine Global N-Back",
        "Routine Instructions Example N-Back",
        "Routine Example N-Back Trial",
        "Routine Stroop_Global",
        "Routine Example Stroop level EASY",
        "Routine Stroop level EASY",
        "Routine Example Stroop level HARD",
        "Routine Stroop level HARD",
        "Routine Sudoku"
    ]

    evaluations_list = [
        "Routine likert Scale",
        "Routine NASA",
        "Routine PANAS",
        "Routine Affective-Slider"
    ]

    all_phrases = [
        "Routine Muse - Psychopy Synchronization",
        "Routine Relaxation-Video",
        "Routine Eye-Closing",
        "Routine Arithmetix",
        "Routine likert Scale",
        "Routine Global N-Back",
        "Routine Instructions Example N-Back",
        "Routine Example N-Back Trial",
        "Routine Stroop_Global",
        "Routine Example Stroop level EASY",
        "Routine Stroop level EASY",
        "Routine Example Stroop level HARD",
        "Routine Stroop level HARD",
        "Routine Sudoku",
        "Routine NASA",
        "Routine PANAS",
        "Routine Affective-Slider"]

    with open(log_path) as f:
        f = f.readlines()

    results_dictionary = {}

    # Starting time of the experiment, useful as the rest of timestamps is relative seconds/minutes/hours, relative to this one!
    start_time_overall = get_timestamp_start_of_experiment_from_log(f)
    results_dictionary['absolute_start_time_overall'] = start_time_overall

    # Timestamps for space bar taps used for synchronization
    synchronization_taps_timestamp_list = get_synchronization_timestamps_from_log(f, check_for_correct_tappings_number)
    results_dictionary['synchronization_taps_timestamp_list'] = synchronization_taps_timestamp_list

    # Relaxation Video info
    relaxation_video_dictionary = get_relaxation_video_from_log(f)
    results_dictionary['relaxation_video_dictionary'] = relaxation_video_dictionary

    # Nasa Results
    nasa_dictionary = get_nasa_results_from_log(f)
    results_dictionary['nasa_dictionary'] = nasa_dictionary
    # print(F"*  Dictionary of all NASA results:\n      {nasa_dictionary}")

    # All Panas results for the whole experiment
    # Use these values as regression labels (1-5). Why are all of them not sliders (float instead of int)
    panas_dictionary = get_panas_results_from_log(f)
    results_dictionary['panas_dictionary'] = panas_dictionary
    # print(F"*  Dictionary of all Panas results:\n      {panas_dictionary}")

    # Affective Slider
    affective_slider_dictionary = get_affective_slider_results_from_log(f)
    results_dictionary['affective_slider_dictionary'] = affective_slider_dictionary
    # print(F"*  Dictionary of all Affective-Slider results:\n      {affective_slider_dictionary}")

    # Likert Scale
    likert_scale_dictionary = get_likert_scale_results_from_log(f)
    results_dictionary['likert_scale_dictionary'] = likert_scale_dictionary
    # print(F"*  Dictionary of all likert scale trial results:\n      {likert_scale_dictionary}")

    # Eye closing baseline
    eye_closing_baseline_dictionary = get_eye_closing_baseline_from_log(f)
    results_dictionary['eye_closing_baseline_dictionary'] = eye_closing_baseline_dictionary
    # print(F"*  Dictionary of eye closing baseline timestamps:\n      {eye_closing_baseline_dictionary}")

    # Arithmtic exercise
    arithmetic_exercise_dictionary = get_arithmetic_results_from_log(f)
    results_dictionary['arithmetic_exercise_dictionary'] = arithmetic_exercise_dictionary
    # print(F"*  Dictionary of all arithmetic trial results:\n      {arithmetic_exercise_dictionary}")

    # N-Back Example Trials and N-Back actual results
    n_back_example_exercise_dictionary, n_back_exercise_dictionary = get_n_back_from_log(f)
    results_dictionary['n_back_example_exercise_dictionary'] = n_back_example_exercise_dictionary
    results_dictionary['n_back_exercise_dictionary'] = n_back_exercise_dictionary

    # Stroop Example Trial
    stroop_example_exercise_dictionary, stroop_exercise_dictionary = get_stroop_from_log(f)
    results_dictionary['stroop_example_exercise_dictionary'] = stroop_example_exercise_dictionary
    results_dictionary['stroop_exercise_dictionary'] = stroop_exercise_dictionary

    # Sudoku Results
    sudoku_exercise_dictionary = get_sudoku_results_from_log(f)
    results_dictionary['sudoku_exercise_dictionary'] = sudoku_exercise_dictionary

    return results_dictionary