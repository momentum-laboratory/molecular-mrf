# Pulse sequence for CEST MRF
The sequences are provided in Bruker `.txt` format and pypulseq `.seq` format. The clinical`.seq` sequence is spin-lock-based. 

Choosing the scanner type of the sequence in `create_seq.py` as `type_s='simulate'` will demonstrate a 3D-snapshot-CEST readout (Mueller et al., MRM 2020).

Choosing `type_s='scanner'` will omit the readout entirely and replace it with a pseudo-ADC block that instructs the scanner to perform its readout as defined by the user.

## Description
The folder contains the protocol for brain imaging: 
- B<sub>1</sub> = 0-4 µT
- T<sub>sat</sub> = 2.56 s
- T<sub>R</sub> = 3.5 s 
- Sat. Pulse: A Gaussian pulse train (tp = 16 ms). 

## Publication
Cohen O, Yu VY, Tringale KR, et al. CEST MR fingerprinting (CEST-MRF) for brain tumor quantification using EPI readout and deep learning reconstruction. Magn Reson Med. 2023; 89: 233-249. doi:10.1002/mrm.29448
