#FA-SIDA
##Learning from Failure: Towards Developing A Disease Diagnosis Assistant that also learns from Unsuccessful Diagnoses

In the last few years, automatic disease diagnosis has gained immense popularity in research and industry communities. Humans learn a task through both successful and unsuccessful attempts in real life, and physicians are not different. When doctors fail to diagnose disease correctly, they re-assess the extracted symptoms and re-diagnose the patient by inspecting a few more symptoms guided by their previous experience and current context. Motivated by the experience gained from failure assessment, we propose a novel end-to-end automatic disease diagnosis dialogue system called Failure Assessment incorporated Symptom Investigation and Disease Diagnosis (FA-SIDD) Assistant. The proposed FA-SIDD model includes a knowledge-guided, incorrect disease projection-aware failure assessment module that analyzes unsuccessful diagnosis attempts and reinforces the assessment for further investigation and re-diagnosis. We formulate a novel Markov decision process for the proposed failure assessment incorporated symptom investigation and disease diagnosis framework and optimize the policy using deep reinforcement learning. The proposed model has outperformed several baselines and the existing symptom investigation and diagnosis methods by a significant margin in all evaluation metrics (including human evaluation). The improvements over the multiple datasets firmly establish the efficacy of learning gained from unsuccessful diagnoses. Furthermore, we developed a conversational medical dialogue corpus called Diagnosis-logue, annotated with utterance-level intent and symptom information in English. To the best of our knowledge, the work is the first attempt towards investigating the importance of learning gained from unsuccessful diagnoses and models the learning in symptom investigation and diagnosis process


## Dataset Settings
- To use SD dataset 
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
