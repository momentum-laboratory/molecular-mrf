# CEST-MRF sequences
This sequence was used in the Protocol manuscript in the dot_prod and deep_reco examples.
The sequences are provided in Bruker `.txt` format and PyPulseq `.seq` format. 
The clinical `.seq` sequence is spin-lock-based, with a fixed readout (for the simulation). Choosing the scanner type of sequence in `create_seq.py` as `type_s='scanner'` will omit the readout entirely and replace it with a pseudo-ADC block that instructs the scanner to perform its readout. 


## Description
The folder contains the protocol for 3.0 ppm (L-arginine) MRF imaging: 
- B<sub>1</sub> = 0-6 µT
- T<sub>sat</sub> = 2.5 s
- T<sub>R</sub> = 3.5 s 
- Pulse shape: a rectangular pulse train with 50% DC (100 ms 'on') for the .seq 3T clinical scanner version. CW for the preclinical version.

## Publication
Cohen, O., Huang, S., McMahon, M. T., Rosen, M. S. & Farrar, C. T. Rapid and quantitative chemicalexchange saturation transfer (CEST) imaging with magnetic resonance fingerprinting (MRF). Magnetic resonance in medicine 80, 2449–2463 (2018) 




