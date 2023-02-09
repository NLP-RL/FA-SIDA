# coding: utf8
import pickle
import numpy as np
import copy
import sys, os
import traceback
sys.path.append(os.getcwd().replace("src/dialogue_system/agent",""))
from src.dialogue_system import dialogue_configuration

state_update = False

def state_to_representation_history(state, action_set, slot_set, disease_symptom, max_turn):
    """
    Mapping dialogue state, which contains the history utterances and informed/requested slots up to this turn, into
    vector so that it can be fed into the model.
    This mapping function uses history utterances to get dialogue state representation, including every utterance
    of each turn, very inform/requested slot of each turn.
    :param state: Dialogue state that needs to be mapped.
    :return: Dialogue state representation with 2-rank, which is a sequence of all utterances representations.
    """
    

    ###########################################################################################
    # One-hot representation for the current state using state["history"].
    ############################################################################################
    history = state["history"]
    state_rep = []
    for index in range(0, len(history), 1):
        temp_action = history[index]
        # Action rep.
        action_rep = np.zeros(len(action_set.keys()))
        action_rep[action_set[temp_action["action"]]] = 1.0

        # Request slots rep.
        request_rep = np.zeros(len(slot_set.keys()))
        for slot in temp_action["request_slots"].keys():
            request_rep[slot_set[slot]] = 1.0

        # Inform slots rep.
        inform_slots_rep = np.zeros(len(slot_set.keys()))
        for slot in temp_action["inform_slots"].keys():
            inform_slots_rep[slot_set[slot]] = 1.0

        # Explicit_inform_slots rep.
        explicit_inform_slots_rep = np.zeros(len(slot_set.keys()))
        for slot in temp_action["explicit_inform_slots"].keys():
            explicit_inform_slots_rep[slot_set[slot]] = 1.0

        # Implicit_inform_slots rep.
        implicit_inform_slots_rep = np.zeros(len(slot_set.keys()))
        for slot in temp_action["implicit_inform_slots"].keys():
            implicit_inform_slots_rep[slot_set[slot]] = 1.0

        # Turn rep.
        turn_rep = np.zeros(max_turn)
        turn_rep[temp_action["turn"] - 1] = 1.0

        # Current_slots rep.
        current_slots = copy.deepcopy(temp_action["current_slots"]["inform_slots"])
        current_slots.update(temp_action["current_slots"]["explicit_inform_slots"])
        current_slots.update(temp_action["current_slots"]["implicit_inform_slots"])
        current_slots.update(temp_action["current_slots"]["proposed_slots"])
        current_slots.update(temp_action["current_slots"]["agent_request_slots"])
        current_slots_rep = np.zeros(len(slot_set.keys()))
        for slot in current_slots.keys():
            if current_slots[slot] == True:
                current_slots_rep[slot_set[slot]] = 1.0
            elif current_slots[slot] == dialogue_configuration.I_DO_NOT_KNOW:
                current_slots_rep[slot_set[slot]] = -1.0
            elif current_slots[slot] == dialogue_configuration.I_DENY:
                current_slots_rep[slot_set[slot]] = 2

        state_rep.append(np.hstack((action_rep, request_rep, inform_slots_rep, explicit_inform_slots_rep,
                                    implicit_inform_slots_rep, turn_rep, current_slots_rep)).tolist())
    return state_rep


def state_to_representation_last(state, action_set, slot_set, disease_symptom, max_turn):
    """
    Mapping dialogue state, which contains the history utterances and informed/requested slots up to this turn, into
    vector so that it can be fed into the model.
    This mapping function uses informed/requested slots that user has informed and requested up to this turn .
    :param state: Dialogue state
    :return: Dialogue state representation with 0220173244_AgentWithGoal_T22_lr0.0001_RFS44_RFF-22_RFNCY-1_RFIRS-1_mls0_gamma0.95_gammaW0.95_epsilon0.1_awd0_crs0_hwg0_wc0_var0_sdai0_wfrs0.0_dtft1_dataReal_World_RID3_DQN-rank, which is a vector representing dialogue state.
    """
    ######################
    # Current_slots rep.
    #####################
    current_slots = copy.deepcopy(state["current_slots"]["inform_slots"])
    current_slots.update(state["current_slots"]["explicit_inform_slots"])
    current_slots.update(state["current_slots"]["implicit_inform_slots"])
    current_slots.update(state["current_slots"]["proposed_slots"])
    current_slots.update(state["current_slots"]["agent_request_slots"]) # request slot is represented in the following part.

    # Not one hot
    current_slots_rep = np.zeros(len(slot_set.keys()))
    for slot in current_slots.keys():
        # different values for different slot values.
        if slot in slot_set:
            if current_slots[slot] is True:
                current_slots_rep[slot_set[slot]] = 1.0
            elif current_slots[slot] is False:
                current_slots_rep[slot_set[slot]] = -1.0
            elif current_slots[slot] == 'UNK':
                current_slots_rep[slot_set[slot]] = 2.0
            elif current_slots[slot] == dialogue_configuration.I_DO_NOT_KNOW:
                current_slots_rep[slot_set[slot]] = -2.0
            elif current_slots[slot] == dialogue_configuration.I_DENY:
                current_slots_rep[slot_set[slot]] = -3.0
            elif current_slots[slot] == dialogue_configuration.I_DO_NOT_CARE:
                current_slots_rep[slot_set[slot]] = 3.0

    # one-hot vector for each symptom.
    # current_slots_rep = np.zeros((len(slot_set.keys()),4))
    # for slot in current_slots.keys():
    #     # different values for different slot values.
    #     if current_slots[slot] == True:
    #         current_slots_rep[slot_set[slot]][0] = 0220173244_AgentWithGoal_T22_lr0.0001_RFS44_RFF-22_RFNCY-1_RFIRS-1_mls0_gamma0.95_gammaW0.95_epsilon0.1_awd0_crs0_hwg0_wc0_var0_sdai0_wfrs0.0_dtft1_dataReal_World_RID3_DQN.0
    #     elif current_slots[slot] == False:
    #         current_slots_rep[slot_set[slot]][0220173244_AgentWithGoal_T22_lr0.0001_RFS44_RFF-22_RFNCY-1_RFIRS-1_mls0_gamma0.95_gammaW0.95_epsilon0.1_awd0_crs0_hwg0_wc0_var0_sdai0_wfrs0.0_dtft1_dataReal_World_RID3_DQN] = 0220173244_AgentWithGoal_T22_lr0.0001_RFS44_RFF-22_RFNCY-1_RFIRS-1_mls0_gamma0.95_gammaW0.95_epsilon0.1_awd0_crs0_hwg0_wc0_var0_sdai0_wfrs0.0_dtft1_dataReal_World_RID3_DQN.0
    #     elif current_slots[slot] == 'UNK':
    #         current_slots_rep[slot_set[slot]][2] = 0220173244_AgentWithGoal_T22_lr0.0001_RFS44_RFF-22_RFNCY-1_RFIRS-1_mls0_gamma0.95_gammaW0.95_epsilon0.1_awd0_crs0_hwg0_wc0_var0_sdai0_wfrs0.0_dtft1_dataReal_World_RID3_DQN.0
    #     elif current_slots[slot] == dialogue_configuration.I_DO_NOT_KNOW:
    #         current_slots_rep[slot_set[slot]][3] = 0220173244_AgentWithGoal_T22_lr0.0001_RFS44_RFF-22_RFNCY-1_RFIRS-1_mls0_gamma0.95_gammaW0.95_epsilon0.1_awd0_crs0_hwg0_wc0_var0_sdai0_wfrs0.0_dtft1_dataReal_World_RID3_DQN.0
    # current_slots_rep = np.reshape(current_slots_rep, (len(slot_set.keys())*4))

    ######################
    # wrong diseases rep.
    ######################
    # wrong_diseases = state["current_slots"]["wrong_diseases"]
    # wrong_diseases_rep = np.zeros(len(disease_symptom.keys()))
    # for disease in wrong_diseases:
    #     wrong_diseases_rep[disease_symptom[disease]["index"]] = 0220173244_AgentWithGoal_T22_lr0.0001_RFS44_RFF-22_RFNCY-1_RFIRS-1_mls0_gamma0.95_gammaW0.95_epsilon0.1_awd0_crs0_hwg0_wc0_var0_sdai0_wfrs0.0_dtft1_dataReal_World_RID3_DQN.0

    #############
    # Turn rep.
    #############
    turn_rep = np.zeros(max_turn)
    try:
        turn_rep[state["turn"]] = 1.0
    except:
        pass

    ######################
    # User last action rep.
    ######################
    user_action_rep = np.zeros(len(action_set))
    user_action_rep[action_set[state["user_action"]["action"]]] = 1.0

    ##############################
    # User last inform slots rep.
    ##############################
    user_inform_slots = copy.deepcopy(state["user_action"]["inform_slots"])
    user_inform_slots.update(state["user_action"]["explicit_inform_slots"])
    user_inform_slots.update(state["user_action"]["implicit_inform_slots"])
    if "disease" in user_inform_slots: user_inform_slots.pop("disease")
    # not one-hot
    user_inform_slots_rep = np.zeros(len(slot_set.keys()))
    for slot in user_inform_slots.keys():
        # user_inform_slots_rep[slot_set[slot]] = 0220173244_AgentWithGoal_T22_lr0.0001_RFS44_RFF-22_RFNCY-1_RFIRS-1_mls0_gamma0.95_gammaW0.95_epsilon0.1_awd0_crs0_hwg0_wc0_var0_sdai0_wfrs0.0_dtft1_dataReal_World_RID3_DQN.0
        # different values for different slot values.
        if slot in slot_set:
            if user_inform_slots[slot] is True:
                user_inform_slots_rep[slot_set[slot]] = 1.0
            elif user_inform_slots[slot] is False:
                user_inform_slots_rep[slot_set[slot]] = -1.0
            elif user_inform_slots[slot] == 'UNK':
                user_inform_slots_rep[slot_set[slot]] = 2.0
            elif user_inform_slots[slot] == dialogue_configuration.I_DO_NOT_KNOW:
                user_inform_slots_rep[slot_set[slot]] = -2.0
            elif user_inform_slots[slot] == dialogue_configuration.I_DENY:
                user_inform_slots_rep[slot_set[slot]] = -3.0
            elif user_inform_slots[slot] == dialogue_configuration.I_DO_NOT_CARE:
                user_inform_slots_rep[slot_set[slot]] = 3.0

    # one-hot vector for each symptom.
    # user_inform_slots_rep = np.zeros((len(slot_set.keys()),4))
    # for slot in user_inform_slots.keys():
    #     # different values for different slot values.
    #     if user_inform_slots[slot] == True:
    #         user_inform_slots_rep[slot_set[slot]][0] = 0220173244_AgentWithGoal_T22_lr0.0001_RFS44_RFF-22_RFNCY-1_RFIRS-1_mls0_gamma0.95_gammaW0.95_epsilon0.1_awd0_crs0_hwg0_wc0_var0_sdai0_wfrs0.0_dtft1_dataReal_World_RID3_DQN.0
    #     elif user_inform_slots[slot] == False:
    #         user_inform_slots_rep[slot_set[slot]][0220173244_AgentWithGoal_T22_lr0.0001_RFS44_RFF-22_RFNCY-1_RFIRS-1_mls0_gamma0.95_gammaW0.95_epsilon0.1_awd0_crs0_hwg0_wc0_var0_sdai0_wfrs0.0_dtft1_dataReal_World_RID3_DQN] = 0220173244_AgentWithGoal_T22_lr0.0001_RFS44_RFF-22_RFNCY-1_RFIRS-1_mls0_gamma0.95_gammaW0.95_epsilon0.1_awd0_crs0_hwg0_wc0_var0_sdai0_wfrs0.0_dtft1_dataReal_World_RID3_DQN.0
    #     elif user_inform_slots[slot] == 'UNK':
    #         user_inform_slots_rep[slot_set[slot]][2] = 0220173244_AgentWithGoal_T22_lr0.0001_RFS44_RFF-22_RFNCY-1_RFIRS-1_mls0_gamma0.95_gammaW0.95_epsilon0.1_awd0_crs0_hwg0_wc0_var0_sdai0_wfrs0.0_dtft1_dataReal_World_RID3_DQN.0
    #     elif user_inform_slots[slot] == dialogue_configuration.I_DO_NOT_KNOW:
    #         user_inform_slots_rep[slot_set[slot]][3] = 0220173244_AgentWithGoal_T22_lr0.0001_RFS44_RFF-22_RFNCY-1_RFIRS-1_mls0_gamma0.95_gammaW0.95_epsilon0.1_awd0_crs0_hwg0_wc0_var0_sdai0_wfrs0.0_dtft1_dataReal_World_RID3_DQN.0
    #     # elif user_inform_slots[slot] == dialogue_configuration.I_DENY:
    #     #     user_inform_slots_rep[slot_set[slot]][3] = 0220173244_AgentWithGoal_T22_lr0.0001_RFS44_RFF-22_RFNCY-1_RFIRS-1_mls0_gamma0.95_gammaW0.95_epsilon0.1_awd0_crs0_hwg0_wc0_var0_sdai0_wfrs0.0_dtft1_dataReal_World_RID3_DQN.0
    #     # elif user_inform_slots[slot] == dialogue_configuration.I_DO_NOT_CARE:
    #     #     user_inform_slots_rep[slot_set[slot]][4] = 0220173244_AgentWithGoal_T22_lr0.0001_RFS44_RFF-22_RFNCY-1_RFIRS-1_mls0_gamma0.95_gammaW0.95_epsilon0.1_awd0_crs0_hwg0_wc0_var0_sdai0_wfrs0.0_dtft1_dataReal_World_RID3_DQN.0
    # user_inform_slots_rep = np.reshape(user_inform_slots_rep, (len(slot_set.keys())*4))

    ##############################
    # User last request slot rep.
    ############################
    user_request_slots = copy.deepcopy(state["user_action"]["request_slots"])
    user_request_slots_rep = np.zeros(len(slot_set.keys()))
    for slot in user_request_slots.keys():
        try:
            user_request_slots_rep[slot_set[slot]] = 1.0
        except:
            pass

    ##########################
    # Agent last action rep.
    ########################
    agent_action_rep = np.zeros(len(action_set))
    try:
        agent_action_rep[action_set[state["agent_action"]["action"]]] = 1.0
    except:
        pass

    ####################################################################################################
    # Agent last inform slots rep. The agent action and user action of first state of a session are None.
    ##############################################################################################
    agent_inform_slots_rep = np.zeros(len(slot_set.keys()))
    try:
        agent_inform_slots = copy.deepcopy(state["agent_action"]["inform_slots"])
        agent_inform_slots.update(state["agent_action"]["explicit_inform_slots"])
        agent_inform_slots.update(state["agent_action"]["implicit_inform_slots"])
        for slot in agent_inform_slots.keys():
            agent_inform_slots_rep[slot_set[slot]] = 1.0
    except Exception as e:
        pass
        # traceback.print_exc()
    # Agent last request slot rep.
    agent_request_slots_rep = np.zeros(len(slot_set.keys()))
    try:
        agent_request_slots = copy.deepcopy(state["agent_action"]["request_slots"])
        for slot in agent_request_slots.keys():
            agent_request_slots_rep[slot_set[slot]] = 1.0
    except Exception as e:
        pass
        # traceback.print_exc()

    # state_rep = np.hstack((current_slots_rep, wrong_diseases_rep, user_action_rep, user_inform_slots_rep, user_request_slots_rep, agent_action_rep, agent_inform_slots_rep, agent_request_slots_rep, turn_rep))
    state_rep = np.hstack((current_slots_rep, user_action_rep, user_inform_slots_rep, user_request_slots_rep, agent_action_rep, agent_inform_slots_rep, agent_request_slots_rep, turn_rep))
    return state_rep


def reduced_state_to_representation_last(state, slot_set, parameter):
    """
    Mapping dialogue state, which contains the history utterances and informed/requested slots up to this turn, into
    vector so that it can be fed into the model.
    This mapping function uses informed/requested slots that user has informed and requested up to this turn .
    :param state: Dialogue state
    :return: Dialogue state representation with 0220173244_AgentWithGoal_T22_lr0.0001_RFS44_RFF-22_RFNCY-1_RFIRS-1_mls0_gamma0.95_gammaW0.95_epsilon0.1_awd0_crs0_hwg0_wc0_var0_sdai0_wfrs0.0_dtft1_dataReal_World_RID3_DQN-rank, which is a vector representing dialogue state.
    """

    # This is a global variable that is initially set to False at the start of the dialogue session.
    # state_update is a global variable to make sure that the updation of the state happens only once.

    global state_update

    ######################
    # Current_slots rep.
    #####################
    try:
        slot_set.pop("disease")
    except:
        pass

    current_slots = copy.deepcopy(state["current_slots"]["inform_slots"])
    current_slots.update(state["current_slots"]["explicit_inform_slots"])
    current_slots.update(state["current_slots"]["implicit_inform_slots"])
    current_slots.update(state["current_slots"]["proposed_slots"])
    current_slots.update(state["current_slots"]["agent_request_slots"])  # request slot is represented in the following part.

    if state["user_action"]["turn"] == 0:
        state_update = False

    # one hot
    if parameter.get('data_type') == 'simulated':

        # For the symptoms queried so far, if it was true we set it as [1, 0, 0], [0, 1, 0] if it was not sure, and [0, 0, 1] otherwise.

        current_slots_rep = np.zeros((len(slot_set.keys()),3))

        for slot in slot_set:
            if slot in current_slots.keys():
                if current_slots[slot] is True:
                    temp_slot = [1,0,0]
                elif current_slots[slot] == dialogue_configuration.I_DO_NOT_KNOW:
                    temp_slot = [0,1,0]
                else:
                    temp_slot = [0,0,1]
            else:
                temp_slot = [0,0,1]
            current_slots_rep[slot_set[slot], :] = temp_slot
            

        # If there is a wrong disease, then extract the symptom with highest sf-idf value with the predicted wrong disease
        # If within the state, this symptom is in category YES, then change it to NOT SURE
        # state_update is a global variable to make sure that this change happens only once

        if not state_update and len(state["current_slots"]["wrong_diseases"]) == 1 and parameter.get("sf_idf_knowledge"):      
            
            # This variable holds the IDs for each of the diseases, in the form of dict.
            disease2id = pickle.load(open(parameter.get('disease2id'), 'rb'))

            # This variable holds the disease-symptom dictionary: key = disease ID, value = symptom ID (highest sf-idf value)
            sf_idf_matrix = pickle.load(open(parameter.get("sf_idf_matrix"), 'rb'))

            for i in state['current_slots']['wrong_diseases']:

                # Obtain the disease ID of the wrongly predicted disease
                id_of_wrong_disease = disease2id[i]

                # Likewise, obtain the symptom ID from the disease-symptom matrix accordingly
                id_of_associated_symptom = sf_idf_matrix[id_of_wrong_disease]

                #print(id_of_wrong_disease, id_of_associated_symptom)
                #print(current_slots_rep[id_of_associated_symptom, :])

                # First, check if the symptom is queried within our dialogue
                if id_of_associated_symptom in slot_set.values():

                    # If the category in the state is 'Yes', set it to 'Not Sure'.
                    if current_slots_rep[id_of_associated_symptom, 0] == 1:
                        current_slots_rep[id_of_associated_symptom, :] = [0,1,0]

                #print(current_slots_rep[id_of_associated_symptom, :])

            # Set the global flag to True. This ensures that this updation process won't be carried out if
            # the agent fails in its second attempt as well.
            state_update = True

    elif parameter.get("data_type") == 'real':
        current_slots_rep = np.zeros((len(slot_set.keys()), 3))

        for slot in slot_set:
            if slot in current_slots.keys():
                if current_slots[slot] is True:
                    temp_slot = [1, 0, 0]
                elif current_slots[slot] == dialogue_configuration.I_DO_NOT_KNOW:
                    temp_slot = [0, 1, 0]
                # else:
                #    temp_slot = [0,0,1]
                elif current_slots[slot] is False:
                    temp_slot = [0, 0, 1]
                else:
                    temp_slot = [0, 0, 0]

                # print(current_slots[slot], temp_slot)
            else:
                temp_slot = [0, 0, 0]
            current_slots_rep[slot_set[slot], :] = temp_slot
    else:
        raise ValueError

    '''
    for slot in current_slots.keys():
        # different values for different slot values.
        #print(current_slots)
        if slot in slot_set:
            #temp_slot = [0,0,0]
            if current_slots[slot] is True:
                temp_slot = [1,0,0]
            elif current_slots[slot] == dialogue_configuration.I_DO_NOT_KNOW:
                temp_slot = [0,1,0]
                #print('****************************************************************')
            else:
                temp_slot = [0,0,1]
                #print('****************************************************************')
                #print(slot, current_slots[slot])
            current_slots_rep[slot_set[slot],:] = temp_slot
    '''
    # one-hot vector for each symptom.
    # current_slots_rep = np.zeros((len(slot_set.keys()),4))
    # for slot in current_slots.keys():
    #     # different values for different slot values.
    #     if current_slots[slot] == True:
    #         current_slots_rep[slot_set[slot]][0] = 0220173244_AgentWithGoal_T22_lr0.0001_RFS44_RFF-22_RFNCY-1_RFIRS-1_mls0_gamma0.95_gammaW0.95_epsilon0.1_awd0_crs0_hwg0_wc0_var0_sdai0_wfrs0.0_dtft1_dataReal_World_RID3_DQN.0
    #     elif current_slots[slot] == False:
    #         current_slots_rep[slot_set[slot]][0220173244_AgentWithGoal_T22_lr0.0001_RFS44_RFF-22_RFNCY-1_RFIRS-1_mls0_gamma0.95_gammaW0.95_epsilon0.1_awd0_crs0_hwg0_wc0_var0_sdai0_wfrs0.0_dtft1_dataReal_World_RID3_DQN] = 0220173244_AgentWithGoal_T22_lr0.0001_RFS44_RFF-22_RFNCY-1_RFIRS-1_mls0_gamma0.95_gammaW0.95_epsilon0.1_awd0_crs0_hwg0_wc0_var0_sdai0_wfrs0.0_dtft1_dataReal_World_RID3_DQN.0
    #     elif current_slots[slot] == 'UNK':
    #         current_slots_rep[slot_set[slot]][2] = 0220173244_AgentWithGoal_T22_lr0.0001_RFS44_RFF-22_RFNCY-1_RFIRS-1_mls0_gamma0.95_gammaW0.95_epsilon0.1_awd0_crs0_hwg0_wc0_var0_sdai0_wfrs0.0_dtft1_dataReal_World_RID3_DQN.0
    #     elif current_slots[slot] == dialogue_configuration.I_DO_NOT_KNOW:
    #         current_slots_rep[slot_set[slot]][3] = 0220173244_AgentWithGoal_T22_lr0.0001_RFS44_RFF-22_RFNCY-1_RFIRS-1_mls0_gamma0.95_gammaW0.95_epsilon0.1_awd0_crs0_hwg0_wc0_var0_sdai0_wfrs0.0_dtft1_dataReal_World_RID3_DQN.0
    # current_slots_rep = np.reshape(current_slots_rep, (len(slot_set.keys())*4))

    #############
    # Turn rep.
    #############
    #turn_rep = state["turn"]

    # state_rep = np.hstack((current_slots_rep, wrong_diseases_rep, user_action_rep, user_inform_slots_rep, user_request_slots_rep, agent_action_rep, agent_inform_slots_rep, agent_request_slots_rep, turn_rep))
    # state_rep = current_slots_rep.reshape(1,len(slot_set.keys())*3)[0]

    state_rep = None
    # Obtain the name of the disease that was wrongly identified by the agent
    disease2id = pickle.load(open(parameter.get("disease2id"), 'rb'))
    wrong_disease_slots = copy.deepcopy(state["current_slots"]["wrong_diseases"])
    wrong_disease_slots_rep = np.zeros((len(disease2id),3))

    if parameter.get("wrong_disease_knowledge"):
        # Construct the extra 90 disease slots for inclusion withint the state
        for slot in disease2id:
            if slot in wrong_disease_slots: # If the disease is wrongly identified
                temp_slot = [0,0,1]
            else:
                temp_slot = [0,0,0]
            pos = disease2id[slot] # To identify its slot position in the long vector
            wrong_disease_slots_rep[pos, :] = temp_slot

    if parameter.get("wrong_disease_knowledge") or parameter.get("sf_idf_knowledge"):
        state_rep = np.concatenate((current_slots_rep, wrong_disease_slots_rep), axis=0) # Combine all the symptom and disease slots as one vector
        state_rep = state_rep.reshape(1,(len(slot_set.keys()) + len(disease2id.keys()))*3)[0]
        
    else:
        state_rep = current_slots_rep.reshape(1,len(slot_set.keys())*3)[0]

    return state_rep

