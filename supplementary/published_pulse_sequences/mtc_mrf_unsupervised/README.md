# Pulse sequence for CEST MRF
The sequences are provided in Bruker `.txt` format and pypulseq `.seq` format. The clinical `.seq` sequence is spin-lock-based, with a fixed readout (for the simulation), which differs from the original paper. 

Note that the original publication used a 100% duty cycle which is not available for all scanner models and vendors. The duty cycle implemented here is 50% (100 ms 'on'). It can be modified as needed using the provided .py files located in this folder.

Choosing the scanner type of sequence in `create_seq.py`, i.e., `type_s='scanner'`, will omit the readout entirely and replace it with a pseudo-ADC that instructs the scanner to perform its readout.

## Description
The folder contains the unsupervised MT protocol for brain imaging: 
- B<sub>1</sub> = 0-2 µT
- T<sub>sat</sub> = 0-2 s
- T<sub>Rec</sub> = 0-5 s 
- ω = 8-50 ppm

## Publication
Kang B, Kim B, Schär M, Park H, Heo HY. Unsupervised learning for magnetization transfer contrast MR fingerprinting: Application to CEST and nuclear Overhauser enhancement imaging. Magn Reson Med. 2021 Apr;85(4):2040-2054. doi: 10.1002/mrm.28573. Epub 2020 Oct 31. PMID: 33128483.
