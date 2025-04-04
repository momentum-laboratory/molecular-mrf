# Global deep learning optimized pulse sequence for CEST MRF
The sequences are provided in Bruker `.txt` format and PyPulseq `.seq` format. The clinical `.seq` sequence is spin-lock-based, with a fixed readout (for the simulation), which does not fully correspond to the publication (the flip angle is not varied here). 

Note that in the original publication the readout flip angle is varied. This can be implemented using the FA provided in the `scone.txt` file (4th column from the left).

Choosing the scanner type of sequence in `create_seq.py`, i.e., `type_s='scanner'`, will omit the readout entirely and replace it with a pseudo-ADC that instructs the scanner to perform its readout. 

## Description
The folder contains the protocol for brain imaging: 
- B<sub>1</sub> = 0-4 µT
- T<sub>sat</sub> = 0-4 s
- T<sub>R</sub> = 0-4 s 
- Sat. Pulse: Gaussian pulse train (16 ms each).

## Publication
Cohen O, Otazo R. Global deep learning optimization of chemical exchange saturation transfer magnetic resonance fingerprinting acquisition schedule. NMR in Biomedicine. 2023; 36(10):e4954. doi:10.1002/nbm.4954
(https://analyticalsciencejournals.onlinelibrary.wiley.com/doi/10.1002/nbm.4954#)
