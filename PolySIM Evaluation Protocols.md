Dear Participants, 

While checking the code submitted by the best-ranking teams in the evaluation phase on CodaBench, we noticed that several teams used Urdu data for training. 

However, we would like to clarify that this does not comply with  the evaluation protocols that all teams \*had to\* enforce for the final evaluation phase.

|  | Training Data |  |  |  | Test Data |  |  |  |
| :---- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- |
|  | English |  | Urdu |  | English (in-language) |  | Urdu (cross-lingual) |  |
|  | Face | Voice | Face | Voice | Face | Voice | Face | Voice |
| P3 In-language multimodal | Yes | Yes | NO | NO | Yes | Yes | NO | NO |
| P4 In-language Missing-modality | Yes | Yes | NO | NO | NO | Yes | NO | NO |
| P5 Cross-lingual multimodal | Yes | Yes | NO | NO | NO | NO | Yes | Yes |
| P6 Cross-linguag Missing modality | Yes | Yes | NO | NO | NO | NO | NO | Yes |

The table above provides a detailed description of which data can be used in each phase (train and test/inference) for each protocol. In particular, notice that training should happen in ENGLISH ONLY for ALL PROTOCOLS (p3, p4, p5, and p6) and evaluation should happen in ENGLISH (in-language) for p3 and p4, and in URDU (cross-lingual) for p5 and p6.

Once this has been clarified, we would like to invite you to resubmit your files for the final evaluation ENSURING TO COMPLY WITH THE PROTOCOL DESCRIBED ABOVE.

To ensure that winning teams are not violating the protocol, we will ask the winning teams to submit their code, and we will check that the produced scores are consistent with the ones submitting BY USING ONLY THE ALLOWED DATA. Any team for which the scores were not reproduced correctly, will be automatically disqualified.

We are hence not considering all past submissions to the evaluation phase and re-opening the evaluation phase for the next 3 days (30th May, 31st May, and 1st June).

All three best-scoring teams according to codabench will HAVE TO submit their code for reproducing the score by June 2nd.

We are grateful for your active participation in the challenge.

Kind regards,   
The Poly-SIM team  
