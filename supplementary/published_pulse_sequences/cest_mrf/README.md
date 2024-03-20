# CEST-MRF sequences
The sequences are provided in Bruker `.txt` format and PyPulseq `.seq` format. The clinical `.seq` sequence is spin-lock-based, with a fixed readout (for the simulation), which does not fully correspond to the publication (the flip angle is not varied here). Choosing the scanner type of sequence in `create_seq.py`, i.e., `type_s='scanner'`, will omit the readout entirely and replace it with a pseudo-ADC that instructs the scanner to perform its readout. The sequences used in the Nat. Protocols dot_prod and deep_reco examples.

## Description
The folder contains the protocol for 3.0 ppm (L-arginine) imaging: 
- B<sub>1</sub> = 0-6 µT
- T<sub>sat</sub> = 2.5 s
- T<sub>R</sub> = 3.5 s 
- Pulse shape: Block 100 ms

## Publication
Perlman, O, Farrar, CT, Heo, H-Y. MR Fingerprinting for Semisolid Magnetization Transfer and Chemical Exchange Saturation Transfer Quantification. NMR in Biomedicine. 2022;. e4710 https://doi.org/10.1002/nbm.4710