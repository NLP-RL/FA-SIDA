# Learning from Failure: Towards Developing A Disease Diagnosis Assistant that also learns from Unsuccessful Diagnoses

- Location of the main file for running the models : 


    Final_LUDV0_Sub/Code/src/dialogue_system/run/run.py

## Dataset Settings
- To use SD dataset :


    dataset_type = "SD"
    dataset_location = "Final_LUDV0_Sub/Code/src/classifier/data/sd_dataset"
- To use MD dataset :


    dataset_type = "MD"
    dataset_location = "Final_LUDV0_Sub/Code/src/classifier/data/md_dataset"

## DQN Algorithm Settings
- To use DQN algorithm : 


    dqn_type = "DQN"
- To use Double DQN algorithm :


    dqn_type = "DoubleDQN"
- To use prioritized experience replay:


    prioritized_replay = True
## Running different models
- To run Flat-DQN :


    agent_id = "agentdqn"
    disease_as_action = True
    use_all_labels = False
    
- To run HRL : 


    agent_id = "agenthrljoint2"
    allow_wrong_disease = False
    wrong_disease_knowledge = False
    sf_idf_knowledge = False
    disease_as_action = False
    classifier_type = "deep_learning"
    use_all_labels = True

- To run FA-SIDA with only UER :


    agent_id = "agenthrljoint2"
    allow_wrong_disease = True
    wrong_disease_knowledge = True
    sf_idf_knowledge = False
	disease_as_action = False
	classifier_type = "deep_learning"
	use_all_labels = True

- To run FA-SIDA with only DS-KG :


    agent_id = "agenthrljoint2"
    allow_wrong_disease = True
    wrong_disease_knowledge = False
    sf_idf_knowledge = True
	disease_as_action = False
	classifier_type = "deep_learning"
	use_all_labels = True

- To run FA-SIDA :


    agent_id = "agenthrljoint2"
    allow_wrong_disease = True
    wrong_disease_knowledge = True
    sf_idf_knowledge = True
	disease_as_action = False
	classifier_type = "deep_learning"
	use_all_labels = True


